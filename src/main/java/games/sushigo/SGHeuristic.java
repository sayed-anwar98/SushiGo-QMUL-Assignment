package games.sushigo;

import core.AbstractGameState;
import core.components.Counter;
import core.interfaces.IStateHeuristic;
import games.sushigo.cards.SGCard.SGCardType;

import java.util.Map;

/**
 * Heuristic for Sushi Go! that values progress towards scoring, not just banked points.
 *
 * Banked points (getGameScore) already include: completed tempura pairs, completed sashimi
 * triples, dumplings and nigiri (with wasabi). What they miss, and this class adds:
 *  - a half-finished tempura pair or sashimi set (weighted by the chance of completing it)
 *  - an unused wasabi that will multiply a later nigiri
 *  - maki rolls: points are paid at round end for the most / second most
 *  - puddings: points are paid at game end for the most / fewest
 *
 * All values are in points, so discountFactor in RHEA still behaves sensibly.
 * The weights below are guesses to be tuned by experiment.
 */
public class SGHeuristic implements IStateHeuristic {

    // Probability-style weights for things that are not yet scored. Tune these.
    public double wTempura = 0.5;   // one tempura waiting for its pair
    public double wSashimi2 = 0.5;  // two sashimi, one more needed
    public double wSashimi1 = 0.25; // one sashimi, two more needed
    public double wWasabi = 0.5;    // unused wasabi finds a nigiri later
    public double wMaki = 0.6;      // maki lead at round end
    public double wPudding = 0.4;   // pudding lead / deficit at game end

    @Override
    public double evaluateState(AbstractGameState gs, int playerId) {
        SGGameState state = (SGGameState) gs;
        double banked = state.getGameScore(playerId);

        // At the end of the game all points (maki, pudding) are already in the score
        if (!state.isNotTerminal()) return banked;

        SGParameters params = (SGParameters) state.getGameParameters();
        double bonus = 0;

        // Half-finished tempura pair
        if (count(state, playerId, SGCardType.Tempura) % 2 == 1)
            bonus += wTempura * params.valueTempuraPair;

        // Partly built sashimi triple
        int sashimiRemainder = count(state, playerId, SGCardType.Sashimi) % 3;
        if (sashimiRemainder == 2) bonus += wSashimi2 * params.valueSashimiTriple;
        else if (sashimiRemainder == 1) bonus += wSashimi1 * params.valueSashimiTriple;

        // Unused wasabi: extra value of the next nigiri (salmon used as a typical nigiri)
        int wasabi = count(state, playerId, SGCardType.Wasabi);
        bonus += wasabi * wWasabi * (params.multiplierWasabi - 1) * params.valueSalmonNigiri;

        // Maki: how many players have strictly more rolls than me
        int myMaki = count(state, playerId, SGCardType.Maki);
        if (myMaki > 0) {
            int ahead = 0;
            for (int p = 0; p < state.getNPlayers(); p++)
                if (p != playerId && count(state, p, SGCardType.Maki) > myMaki) ahead++;
            if (ahead == 0) bonus += wMaki * params.valueMakiMost;
            else if (ahead == 1) bonus += wMaki * params.valueMakiSecond;
        }

        // Pudding: persists across rounds, scored at game end
        int myPudding = count(state, playerId, SGCardType.Pudding);
        int most = myPudding, least = myPudding;
        boolean anyoneBehind = false, anyoneAhead = false;
        for (int p = 0; p < state.getNPlayers(); p++) {
            if (p == playerId) continue;
            int c = count(state, p, SGCardType.Pudding);
            if (c > myPudding) anyoneAhead = true;
            if (c < myPudding) anyoneBehind = true;
            most = Math.max(most, c);
            least = Math.min(least, c);
        }
        if (!anyoneAhead && anyoneBehind && myPudding > 0)
            bonus += wPudding * params.valuePuddingMost;
        if (state.getNPlayers() > 2 && myPudding == least && anyoneAhead)
            bonus += wPudding * params.valuePuddingLeast; // value is negative

        return banked + bonus;
    }

    private static int count(SGGameState state, int player, SGCardType type) {
        Map<SGCardType, Counter> played = state.getPlayedCardTypes()[player];
        Counter c = played.get(type);
        return c == null ? 0 : c.getValue();
    }
}
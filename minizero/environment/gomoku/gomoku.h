#pragma once

#include "base_env.h"
#include "sgf_loader.h"
#include <vector>

namespace minizero::env::gomoku {

const std::string kGomokuName = "gomoku";
const int kGomokuNumPlayer = 2;
const int kGomokuBoardSize = 15;

class GomokuAction : public BaseAction {
public:
    GomokuAction() : BaseAction() {}
    GomokuAction(int action_id, Player player) : BaseAction(action_id, player) {}
    GomokuAction(const std::vector<std::string>& action_string_args);

    inline Player nextPlayer() const override { return getNextPlayer(player_, kGomokuNumPlayer); }
    inline std::string toConsoleString() const override { return minizero::utils::SGFLoader::actionIDToBoardCoordinateString(getActionID(), kGomokuBoardSize); }
};

class GomokuEnv : public BaseEnv<GomokuAction> {
public:
    GomokuEnv() {}

    void reset() override;
    bool act(const GomokuAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<GomokuAction> getLegalActions() const override;
    bool isLegalAction(const GomokuAction& action) const override;
    bool isTerminal() const override;
    float getEvalScore(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::string toString() const override;
    inline std::string name() const override { return kGomokuName; }

private:
    Player updateWinner(const GomokuAction& action);
    int calculateNumberOfConnection(int start_pos, std::pair<int, int> direction);
    std::string getCoordinateString() const;

    Player winner_;
    std::vector<Player> board_;
};

class GomokuEnvLoader : public BaseEnvLoader<GomokuAction, GomokuEnv> {
public:
    inline std::vector<float> getActionFeatures(int id, utils::Rotation rotation = utils::Rotation::kRotationNone) const override
    {
        assert(id < static_cast<int>(action_pairs_.size()));
        std::vector<float> action_features(kGomokuBoardSize * kGomokuBoardSize, 0.0f);
        action_features[getRotatePosition(action_pairs_[id].first.getActionID(), rotation)] = 1.0f;
        return action_features;
    }

    inline int getPolicySize() const override { return kGomokuBoardSize * kGomokuBoardSize; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return getPositionByRotating(rotation, position, kGomokuBoardSize); }
    inline std::string getEnvName() const override { return kGomokuName; }
};

} // namespace minizero::env::gomoku
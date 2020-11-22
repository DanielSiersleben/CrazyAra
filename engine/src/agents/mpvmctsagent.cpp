#include "mpvmctsagent.h"
#ifdef MPV_MCTS
MPVMCTSAgent::MPVMCTSAgent(NeuralNetAPI* smallNetSingle,
                           NeuralNetAPI* largeNetSingle,
                           vector<unique_ptr<NeuralNetAPI>>& netBatches,
                           vector<unique_ptr<NeuralNetAPI>>& mpvNetBatches,
                           SearchSettings* searchSettings,
                           PlaySettings* playSettings):
    MCTSAgent(smallNetSingle, largeNetSingle, netBatches, searchSettings, playSettings)
{
    for (auto i = 0; i < searchSettings->mpvThreads; ++i) {
        searchThreads.emplace_back(new SearchThread(mpvNetBatches[i].get(), searchSettings, &mapWithMutex));
    }
}
#endif

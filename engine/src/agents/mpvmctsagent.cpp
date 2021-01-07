#include "mpvmctsagent.h"
#include "mpvsearchthread.h"

#ifdef MPV_MCTS
MPVMCTSAgent::MPVMCTSAgent(NeuralNetAPI* smallNetSingle,
                           vector<unique_ptr<NeuralNetAPI>>& netBatches,
                           vector<unique_ptr<NeuralNetAPI>>& mpvNetBatches,
                           SearchSettings* searchSettings,
                           PlaySettings* playSettings):
    MCTSAgent(smallNetSingle, netBatches, searchSettings, playSettings)
{
    for (auto i = 0; i < searchSettings->mpvThreads; ++i) {
        searchThreads.emplace_back(new MPVSearchThread(mpvNetBatches[i].get(), searchSettings, &mapWithMutex, &largeNetNodeQueue));
    }
}
MPVMCTSAgent::~MPVMCTSAgent(){};
#endif

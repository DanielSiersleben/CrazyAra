#ifndef MPVMCTSAGENT_H
#define MPVMCTSAGENT_H
#ifdef MPV_MCTS

#include "mctsagent.h"
#include "agent.h"
#include "../evalinfo.h"
#include "../node.h"
#include "../stateobj.h"
#include "../nn/neuralnetapi.h"
#include "config/searchsettings.h"
#include "config/searchlimits.h"
#include "config/playsettings.h"
#include "../searchthread.h"
#include "../manager/timemanager.h"
#include "../manager/threadmanager.h"
#include "util/gcthread.h"

using namespace crazyara;

class MPVMCTSAgent : public MCTSAgent
{
public:
    MPVMCTSAgent(NeuralNetAPI* smallNetSingle,
              vector<unique_ptr<NeuralNetAPI>>& netBatches,
              vector<unique_ptr<NeuralNetAPI>>& mpvNetBatches,
              SearchSettings* searchSettings,
              PlaySettings* playSettings);
    MPVMCTSAgent(const MPVMCTSAgent&) = delete;
    MPVMCTSAgent& operator=(MPVMCTSAgent const&) = delete;
};

#endif
#endif // MPVMCTSAGENT_H

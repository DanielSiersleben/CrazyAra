#ifndef MPVSEARCHTHREAD_H
#define MPVSEARCHTHREAD_H
#ifdef MPV_MCTS

#include "searchthread.h"
#include "mpvnodequeue.h"
#include <queue>

class MPVSearchThread : public SearchThread
{
public:
    MPVSearchThread(NeuralNetAPI* netBatch, SearchSettings* searchSettings, MapWithMutex* mapWithMutex, MPVNodeQueue* nodeQueue);

    void create_mini_batch() override;

    void thread_iteration() override;

    void reset_stats() override;

    void backup_value_outputs() override;

    void backup_mpvnet_values(FixedVector<Node*>* nodes, vector<Trajectory>& trajectories);

private:
    void set_nn_results_to_child_nodes() override;
};

void fill_mpvnn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, Node *node, size_t& tbHits, SideToMove sideToMove, const SearchSettings* searchSettings);


#endif
#endif // MPVSEARCHTHREAD_H

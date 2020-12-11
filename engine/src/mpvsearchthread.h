#ifndef MPVSEARCHTHREAD_H
#define MPVSEARCHTHREAD_H
#ifdef MPV_MCTS

#include "searchthread.h"
#include "mpvnodequeue.h"
#include <queue>

class MPVSearchThread : public SearchThread
{
private:
    thread** workerThreads;
public:
    MPVSearchThread(NeuralNetAPI* netBatch, SearchSettings* searchSettings, MapWithMutex* mapWithMutex, MPVNodeQueue* nodeQueue);

    void create_mpv_mini_batch();

    void thread_iteration() override;

    void reset_stats() override;

    void backup_value_outputs() override;

    void deleteWorkerThreads();

    void set_is_running(bool value) override;

private:
    void set_nn_results_to_child_nodes() override;
};

void fill_mpvnn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, Node *node, size_t& tbHits, SideToMove sideToMove, const SearchSettings* searchSettings);

void backup_mpvnet_values(FixedVector<Node*>* nodes, vector<Trajectory>* trajectories, atomic_int* idx, SearchSettings* searchSettings);

#endif
#endif // MPVSEARCHTHREAD_H

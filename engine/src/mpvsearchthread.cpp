#include "mpvsearchthread.h"
#include <thread>


#ifdef MPV_MCTS

MPVSearchThread::MPVSearchThread(NeuralNetAPI* netBatch, SearchSettings* searchSettings, MapWithMutex* mapWithMutex, MPVNodeQueue *nodeQueue):
    SearchThread(netBatch, searchSettings, mapWithMutex, nodeQueue, true)
{
     newNodes = make_unique<FixedVector<Node*>>(searchSettings->largeNetBatchSize);
     newNodeSideToMove = make_unique<FixedVector<SideToMove>>(searchSettings->largeNetBatchSize);

     nodeQueue->setInputPlanesAndBuffer(inputPlanes, inputBuffer);
     this->workerThreads = new thread*[searchSettings->largeNetBackpropThreads];
}
MPVSearchThread::~MPVSearchThread(){
    delete[] workerThreads;
}

void MPVSearchThread::create_mpv_mini_batch()
{
    nodeQueue->lock();

    Node** tmp_nodes = nodeQueue->getQueue();
    SideToMove* tmp_sideToMove = nodeQueue->getSideToMove();
    Trajectory* tmp_trajectories = nodeQueue->getTrajectories();
       for(auto i = 0; i < searchSettings->largeNetBatchSize; ++i){
           newNodes->add_element(tmp_nodes[i]);
           newNodeSideToMove->add_element(tmp_sideToMove[i]);
           newTrajectories.emplace_back(tmp_trajectories[i]);
       }
}

void MPVSearchThread::set_nn_results_to_child_nodes()
{
    size_t batchIdx = 0;
    for (auto node: *newNodes) {
        if (!node->is_terminal()) {
            fill_mpvnn_results(batchIdx, net->is_policy_map(), valueOutputs, probOutputs, node, tbHits, newNodeSideToMove->get_element(batchIdx), searchSettings);
        }
        ++batchIdx;
    }
}

void MPVSearchThread::thread_iteration()
{
    if(nodeQueue->batch_is_ready()){
        create_mpv_mini_batch();

        net->predict(nodeQueue->getInputBuffer(), valueOutputs, probOutputs);
        nodeQueue->unlock();

        set_nn_results_to_child_nodes();

        // now all buffers can be reused
        nodeQueue->mark_batch_completed();
        backup_value_outputs();
        newNodeSideToMove->reset_idx();
    }
}

void backup_mpvnet_values(FixedVector<Node*>* nodes, const vector<Trajectory>& trajectories, atomic_int* idx, SearchSettings* searchSettings)
{
    int i;
    while((i = idx->fetch_add(1)) < nodes->size()){
        const Node* node = nodes->get_element(i);
        backup_mpv_value(node->get_value(), trajectories[i], searchSettings->largeNetEvalThreshold, searchSettings->resetQVal);
    }
}

void MPVSearchThread::backup_value_outputs()
{
    const vector<Trajectory>& trajectories = newTrajectories;
    if(searchSettings->largeNetValueBackprop){
        if(searchSettings->largeNetBackpropThreads > 1){
            atomic_int idx = 0;
            for(auto i = 0; i < searchSettings->largeNetBackpropThreads; ++i){
                workerThreads[i] = new thread(backup_mpvnet_values, newNodes.get(), trajectories, &idx, searchSettings);
            }

            for(auto i = 0; i < searchSettings->largeNetBackpropThreads; ++i){
                workerThreads[i]->join();
            }
        }
        else{
            for(size_t idx = 0; idx < newNodes->size(); ++idx){
                    const Node* node = newNodes->get_element(idx);
                    backup_mpv_value(node->get_value(), trajectories[idx], searchSettings->largeNetEvalThreshold, searchSettings->resetQVal);
            }
        }
    }

    newNodes->reset_idx();
    newTrajectories.clear();
}

void MPVSearchThread::set_is_running(bool value)
{
    isRunning = value;
    if(!value){
        nodeQueue->mark_nodes_as_dequeued();
    }
}

void fill_mpvnn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, Node *node, size_t& tbHits, SideToMove sideToMove, const SearchSettings* searchSettings)
{
    node->set_probabilities_for_moves(get_policy_data_batch(batchIdx, probOutputs, isPolicyMap), sideToMove);
    node_post_process_policy(node, searchSettings->nodePolicyTemperature, isPolicyMap, searchSettings);

    (searchSettings->resetQVal || !node->has_nn_results()) ? node->set_mpv_value<true>(valueOutputs[batchIdx], searchSettings->largeNetEvalThreshold) :
    node->set_mpv_value<false>(valueOutputs[batchIdx], searchSettings->largeNetEvalThreshold);

    //node_assign_value(node, valueOutputs, tbHits, batchIdx);

    node->enable_has_large_nn_results();
}

#endif

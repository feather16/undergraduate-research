#include "wl_kernel.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;

int wl_kernel_c_(const std::vector<int>& cell1, const std::vector<int>& cell2, const int H){
    const vector<vector<int>> NEXT_NODES = {
        {1, 2, 4},
        {3, 5},
        {6},
        {6},
        {7},
        {7},
        {7}
    };
    constexpr int N_NEXT_NODES[] = {3, 2, 1, 1, 1, 1, 1};
    const string LABEL_MAX_S = "6";
    constexpr int N_NODES = 8;
    
    vector<string> cell_labels1;
    vector<string> cell_labels2;
    cell_labels1.reserve((N_NODES - 1) * (H + 1));
    cell_labels2.reserve((N_NODES - 1) * (H + 1));
    for(int i = 0; i < N_NODES - 1; i++){
        cell_labels1.push_back({(char)(cell1[i] + '0')});
        cell_labels2.push_back({(char)(cell2[i] + '0')});
    }

    for(int h = 0; h < H; h++){
        const int s_index = (N_NODES - 1) * h;

        for(int i = 0; i < N_NODES - 1; i++){
            vector<string> nb_labels;
            nb_labels.reserve(N_NEXT_NODES[i]);
            for(const int j : NEXT_NODES[i]){
                nb_labels.push_back(j != N_NODES - 1 ? cell_labels1[s_index + j] : LABEL_MAX_S);
            }
            sort(nb_labels.begin(), nb_labels.end());
            string tmp = {(char)('a' - 1 + N_NEXT_NODES[i])};
            for(int k = 0; k < (int)nb_labels.size(); k++){
                tmp += nb_labels[k];
            }
            tmp += cell_labels1[s_index + i];
            cell_labels1.push_back(tmp);
        }

        for(int i = 0; i < N_NODES - 1; i++){
            vector<string> nb_labels;
            nb_labels.reserve(N_NEXT_NODES[i]);
            for(const int j : NEXT_NODES[i]){
                nb_labels.push_back(j != N_NODES - 1 ? cell_labels2[s_index + j] : LABEL_MAX_S);
            }
            sort(nb_labels.begin(), nb_labels.end());
            string tmp = {(char)('a' - 1 + N_NEXT_NODES[i])};
            for(int k = 0; k < (int)nb_labels.size(); k++){
                tmp += nb_labels[k];
            }
            tmp += cell_labels2[s_index + i];
            cell_labels2.push_back(tmp);
        }
    }

    int ret = 1;

    unordered_map<string, int> counter1;

    for(const string& cell_label : cell_labels1){
        const auto itr = counter1.find(cell_label);
        if(itr != counter1.end()){ // ある場合
            itr->second++;
        }
        else{ // ない場合
            counter1[cell_label] = 1;
            //cout<<cell_label<<endl;
        }
    }
    const auto counter1_end = counter1.end();
    for(const string& cell_label : cell_labels2){
        const auto itr = counter1.find(cell_label);
        if(itr != counter1_end){ // ある場合
            ret += itr->second;
        }
    }

    return ret;
}
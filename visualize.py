import matplotlib.pyplot as plt
import pandas as pd
import shelve

def boxplot(data_path):
    # load is_correct and uncertainty scores
    result_df = pd.read_csv(data_path)

    # whole level
    max_prob_P = result_df[result_df['is_wrong'] == 0]['whole_max_prob']
    max_prob_N = result_df[result_df['is_wrong'] == 1]['whole_max_prob']
    avg_prob_P = result_df[result_df['is_wrong'] == 0]['whole_avg_prob']
    avg_prob_N = result_df[result_df['is_wrong'] == 1]['whole_avg_prob']
    max_ent_P = result_df[result_df['is_wrong'] == 0]['whole_max_ent']
    max_ent_N = result_df[result_df['is_wrong'] == 1]['whole_max_ent']
    avg_ent_P = result_df[result_df['is_wrong'] == 0]['whole_avg_ent']
    avg_ent_N = result_df[result_df['is_wrong'] == 1]['whole_avg_ent']
    points = (max_prob_P, max_prob_N, avg_prob_P, avg_prob_N, max_ent_P, max_ent_N, avg_ent_P, avg_ent_N)
    fig, ax = plt.subplots(1, 2, figsize=(24, 16))
    ax[0].boxplot(points)
    ax[0].set_xticklabels(['max_prob_P', 'max_prob_N', 'avg_prob_P', 'avg_prob_N', 'max_ent_P', 'max_ent_N', 'avg_ent_P', 'avg_ent_N'])
    ax[0].set_title('is_wrong to uncertainty scores (whole level)')
    ax[0].set_ylim(0, 4)
    ax[0].grid()

    # thought level
    max_prob_P = result_df[result_df['is_wrong'] == 0]['thought_max_prob']
    max_prob_N = result_df[result_df['is_wrong'] == 1]['thought_max_prob']
    avg_prob_P = result_df[result_df['is_wrong'] == 0]['thought_avg_prob']
    avg_prob_N = result_df[result_df['is_wrong'] == 1]['thought_avg_prob']
    max_ent_P = result_df[result_df['is_wrong'] == 0]['thought_max_ent']
    max_ent_N = result_df[result_df['is_wrong'] == 1]['thought_max_ent']
    avg_ent_P = result_df[result_df['is_wrong'] == 0]['thought_avg_ent']
    avg_ent_N = result_df[result_df['is_wrong'] == 1]['thought_avg_ent']
    points = (max_prob_P, max_prob_N, avg_prob_P, avg_prob_N, max_ent_P, max_ent_N, avg_ent_P, avg_ent_N)
    ax[1].boxplot(points)
    ax[1].set_xticklabels(['max_prob_P', 'max_prob_N', 'avg_prob_P', 'avg_prob_N', 'max_ent_P', 'max_ent_N', 'avg_ent_P', 'avg_ent_N'])
    ax[1].set_title('is_wrong to uncertainty scores (thought level)')
    ax[1].set_ylim(0, 4)
    ax[1].grid()
    
    plt.savefig("boxplot_whole_and_thought.png")
    plt.show()
    print("boxplot generated")


# uncertainty score on each token
def token_score_plot(data_path):
    print(f"Loading {data_path}...")
    shelve_raw_data = shelve.open(data_path)
    raw_data = dict(shelve_raw_data)
    shelve_raw_data.close()
    print(f"Finished loading {data_path}")

    positive_cnt = 0
    negative_cnt = 0
    fig, ax = plt.subplots(20, 2, figsize=(20, 80))
    for key in raw_data:
        if raw_data[key]['is_correct']==1 and positive_cnt<20:
            words = raw_data[key]['decoded_word'][0]
            gen_probs = raw_data[key]['gen_probs'].flatten()
            ax[positive_cnt][0].bar(words, gen_probs)
            ax[positive_cnt][0].set_title(f"scores (instance {int(key)}) [positive]")
            ax[positive_cnt][0].set_ylim(0, 1)
            print(f"\ninstance {int(key)} [positive]:\n{raw_data[key]['decoded_text'][0]}\n")
            # plt.tight_layout()
            positive_cnt += 1
        elif raw_data[key]['is_correct']==0 and negative_cnt<20:
            words = raw_data[key]['decoded_word'][0]
            gen_probs = raw_data[key]['gen_probs'].flatten()
            ax[negative_cnt][1].bar(words, gen_probs)
            ax[negative_cnt][1].set_title(f"chain score (instance {int(key)}) [negative]")
            ax[negative_cnt][1].set_ylim(0, 1)
            print(f"\ninstance {int(key)} [negative]:\n{raw_data[key]['decoded_text'][0]}\n")
            # plt.tight_layout()
            negative_cnt += 1
        if positive_cnt == 20 and negative_cnt == 20:
            print(f"positive_cnt: {positive_cnt}")
            print(f"negative_cnt: {negative_cnt}")
            break
    
    plt.savefig(f"token_score_plot.png")
    plt.show()

    # for i in range(NUM_INSTANCES):
    #     fig, ax = plt.subplots(4,1,figsize=(6,12))
    #     fig.suptitle('uncertainty score of each chain')
    #     chains = []
    #     values_max_prob, values_avg_prob, values_max_ent, values_avg_ent = [], [], [], []
    #     if result_df['is_correct'][i] == 0:
    #         chains = baseline_map[str(i)]['sep_sentence']
    #         print(f"\nchains:\n\n{chains}\n")
    #         values_max_prob = baseline_map[str(i)]['doc_max_prob_foreach_chain']
    #         print(f"\nvalues_max_prob: {values_max_prob}\n")
    #         values_avg_prob = baseline_map[str(i)]['doc_average_prob_foreach_chain']
    #         values_max_ent = baseline_map[str(i)]['doc_max_ent_foreach_chain']
    #         values_avg_ent = baseline_map[str(i)]['doc_average_ent_foreach_chain']
    #         ax[0].barh(chains, values_max_prob)
    #         ax[0].set_title('Max_Prob')
    #         ax[0].set_xlim(0, 5)
    #         ax[1].barh(chains, values_avg_prob)
    #         ax[1].set_title('Avg_Prob')
    #         ax[1].set_xlim(0, 5)
    #         ax[2].barh(chains, values_max_ent)
    #         ax[2].set_title('Max_Ent')
    #         ax[2].set_xlim(0, 5)
    #         ax[3].barh(chains, values_avg_ent)
    #         ax[3].set_title('Avg_Ent')
    #         ax[3].set_xlim(0, 5)
    #         plt.tight_layout()
    #         plt.savefig(f'test_chain_instance{i}.png')
    #         plt.show()

if __name__ == '__main__':
    boxplot(data_path="result_llama.csv")
    # token_score_plot(data_path="raw_data_llama")
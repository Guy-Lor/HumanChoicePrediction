import numpy as np
import json
import pandas as pd
import pickle
################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2

# df_per_hotel_reviews_scores = pd.read_csv('/home/student/project/HumanChoicePrediction/RunningScripts/df_per_hotel_reviews_scores.csv')
with open('/home/student/project/HumanChoicePrediction/data/data_dict.pkl', 'rb') as f:
    dict_loaded_results = pickle.load(f)

################################

def sentiment_ratio_based(ratio_threshold = 2.8469387755102042 ):
    def func(information):
        review_id = information['review_id']
        review_good_bad_sentiment_ratio = dict_loaded_results['positive_to_negative_sentiment_ratio'][review_id]
        if  review_good_bad_sentiment_ratio>=ratio_threshold:
            return 1
        else:
            return 0
    return func

def length_ratio_based(ratio_threshold = 0.8163265306122449):
    def func(information):
        review_id = information['review_id']
        review_good_bad_length_ratio = dict_loaded_results['positive_to_negative_length_ratio'][review_id]
        if  review_good_bad_length_ratio>=ratio_threshold:
            return 1
        else:
            return 0
    return func
###############################

def user_prefered_hotel_metrices(user_vec, threshold=0.0):

    def func(information):
        review_id = information['review_id']
        hotel_metrices_vector = dict_loaded_results['hotel_metrices_vector'][review_id]
        sum_vec = user_vec @ hotel_metrices_vector
        if  sum_vec > threshold:
            return 1
        else:
            return 0
        
    return func

def correct_action(information):
    if information["hotel_value"] >= 8:
        return 1
    else:
        return 0


def random_action(information):
    return np.random.randint(2)


def user_rational_action(information):
    if information["bot_message"] >= 8:
        return 1
    else:
        return 0


def user_picky(information):
    if information["bot_message"] >= 9:
        return 1
    else:
        return 0


def user_sloppy(information):
    if information["bot_message"] >= 7:
        return 1
    else:
        return 0


def user_short_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
            or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def user_picky_short_t4t(information):
    if information["bot_message"] >= 9 or ((information["bot_message"] >= 8) and (
            len(information["previous_rounds"]) == 0 or (
            information["previous_rounds"][-1][REVIEWS].mean() >= 8))):  # good hotel
        return 1
    else:
        return 0


def user_hard_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def history_and_review_quality(history_window, quality_threshold):
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][
                                    -history_window:]])) == 1:  # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0
    return func


def topic_based(positive_topics, negative_topics, quality_threshold):
    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"].loc[topic])*2/(rank+1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"].loc[topic])*2/(rank+1)
        if review_personal_score >= quality_threshold:  # good hotel from user's perspective
            return 1
        else:
            return 0
    return func


def LLM_based(is_stochastic=False, is_high_range_stochastic=False):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    if is_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.rand() <= review_llm_score)
        return func
    
    elif is_high_range_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.uniform(0.5,0.9) <= review_llm_score)
        return func
        
    else:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(review_llm_score >= 0.5)
        return func
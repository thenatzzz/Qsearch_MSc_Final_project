from HELPER_FUNCTION import *


def get_model_each_episode(whole_data):

    # count = 0
    # temp_array = []
    # final_array = []
    # for index_line in range(len(whole_data)):
    #
    #     if len(whole_data[index_line]) != 0 and \
    #         whole_data[index_line][0] == "+++++++++ THERE IS A MATCH !! +++++++++++++++++":
    #
    #         if whole_data[index_line+5][0] == "+++++++++ THERE IS A MATCH !! +++++++++++++++++":
    #             continue
    #
    #         flag = whole_data[index_line]
    #         model_number = whole_data[index_line+1]
    #         accuracy = whole_data[index_line+3]
    #         model = whole_data[index_line+4]
    #         episode_number = whole_data[index_line+5]
    #         count += 1
    #         temp_array.append([flag,model_number,accuracy,model,episode_number])
    #         final_array.append(temp_array)
    #         temp_array = []
    #         continue
    #         print('\n')
    #
    #     if len(whole_data[index_line]) != 0 and \
    #         whole_data[index_line][0] == "_________________ CANNOT FIND A MATCH _______________":
    #         flag = whole_data[index_line]
    #         model_number = whole_data[index_line+29]
    #         accuracy = 'x'
    #         model = 'x'
    #         episode_number = count
    #         count += 1
    #         temp_array.append([flag,model_number,accuracy,model,episode_number])
    #         final_array.append(temp_array)
    #         temp_array =[]
    #
    # print(final_array)
    return 1

def main():
    log_file = "test_qlearning.log"
    data = get_data_from_csv(log_file)
    get_model_each_episode(data)
    # print(data)
    # print(data[0])
if "__main__" == __name__:
    main()

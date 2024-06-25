def check_list(lst, yellow_list=[], blue_list=[]):
    result = "["
    yellow_num = 0
    blue_num = 0
    for i in range(len(lst)):
        item = lst[i]
        if item in yellow_list and item in blue_list:
            result += '?%s? ' % item
            yellow_num, blue_num = yellow_num + 1, blue_num + 1
        elif item in yellow_list:
            result += '|%s| ' % item
            yellow_num = yellow_num + 1
        elif item in blue_list:
            result += '{%s} ' % item
            blue_num = blue_num + 1
        else:
            result += '%s ' % item
        if i % 5 == 4:
            result += '\n'
    result += "]\n"
    result += "num of || = %d, num of {} = %d" % (yellow_num, blue_num)
    return result

def check_tuple_list(lst, yellow_list=[], blue_list=[]):
    def to_string(tp: tuple):
        string = "("
        for t in tp[:-1]:
            string += "%s," % t
        string += "%s)" % tp[-1]
        return string


    result = "["
    yellow_num = 0
    blue_num = 0
    for i in range(len(lst)):
        item = lst[i]
        if item in yellow_list and item in blue_list:
            result += '?%s? ' % to_string(item)
            yellow_num, blue_num = yellow_num + 1, blue_num + 1
        elif item in yellow_list:
            result += '|%s| ' % to_string(item)
            yellow_num = yellow_num + 1
        elif item in blue_list:
            result += '{%s} ' % to_string(item)
            blue_num = blue_num + 1
        else:
            result += '%s ' % to_string(item)
        if i % 5 == 4:
            result += '\n'
    result += "]\n"
    result += "num of || = %d, num of {} = %d" % (yellow_num, blue_num)
    return result
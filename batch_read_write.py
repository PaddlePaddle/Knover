
# 读取program文件，如果里面一行包含了我们需要去除的关键词，就删除这行
def read_write(key_words=["c_sync_calc_stream", "c_sync_comm_stream", "c_broadcast"]):
    contents = open("src_program.txt", "r")
    output_file = open("src_compare_post_clear.txt", "w")
    flag = True
    for line in contents:
        for key_word in key_words:
            if key_word in line:
                flag = False
        if flag:
            output_file.write(line)
        flag = True

read_write()
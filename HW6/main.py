# const
col_company = 0
col_description = 2
col_raise = 6
col_round = 7
data_file_name = '/HW6/dataset'
result_file_name = '/HW6/result'
round_a = 'A'
delimiter = ';'

# program
data = (line for line in open(data_file_name))
next(data)

all_raised = 0
for i in data:
    str_data = i.split(delimiter)
    if ''.join(e for e in str_data[col_round].upper() if e.isalnum()) != round_a:
        continue
    all_raised = all_raised + int(str_data[col_raise])
weight_raised = all_raised / (30e6 - 10e6) * 1000
data = (line for line in open(data_file_name))

result_file = open(result_file_name, 'w')
next(data)
for i in data:
    str_data = i.split(delimiter)
    if ''.join(e for e in str_data[col_round].upper() if e.isalnum()) == round_a\
            and int(str_data[col_raise]) < weight_raised:
        result = str_data[col_company] + delimiter + str_data[col_description] + delimiter + str_data[col_raise] + '\n'
        result_file.write(result)

exit()
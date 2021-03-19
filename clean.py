from const import DATA_FILE_NAME, CLEAN_DATA_FILE_NAME

with open(DATA_FILE_NAME, 'r') as data:
    with open(CLEAN_DATA_FILE_NAME, 'w') as clean_data:
        lines = data.readlines()
        clean_data.write(lines[1])
        for line in lines[3:-1]:
            clean_data.write(line)

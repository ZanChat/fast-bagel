import glob
import json
import os

import pyarrow.parquet as pq

def dump_pq_info(data_dir_list, out_dir):
    out_data = {}
    for data_dir in data_dir_list:
        file_list = glob.glob(os.path.join(data_dir, '*.parquet'))
        for file_path in file_list:
            pq_file = pq.ParquetFile(file_path)
            num_row_groups = pq_file.num_row_groups
            out_data.update({
                file_path :{
                    'num_row_groups': num_row_groups,
                    'num_rows': pq_file.metadata.num_rows,
                    'num_columns': pq_file.metadata.num_columns
                }
            })
            pq_file.close()
    json.dump(out_data, open(out_dir, 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    dump_pq_info(['/some/path/SEED-Data-Edit-Part2-3/bagel_parquet'], '/some/path/SEED-Data-Edit-Part2-3/bagel_parquet_info.json')
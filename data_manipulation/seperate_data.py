import os
import random

random_numbers = random.sample(range(1, 501), 100)

main_path = "D:/Thesis/datasets/BrG_Dataset_Download/"
test_main_path = "D:/Thesis/datasets/test_data/"

for sub_path in os.listdir(main_path):
    t_path = test_main_path+sub_path+"/"
    sub_path = main_path+sub_path+"/"

    if os.path.isdir(sub_path):
        for s_path in os.listdir(sub_path):
            test_path = t_path+s_path+"/"
            s_path = sub_path+s_path+"/"

            if os.path.isdir(s_path):
                src_path = ""
                for path in os.listdir(s_path):
                    if(path.endswith(".png")):
                        src_pa = s_path + path.split()[0]
                    path = path.split()
                    src_path = s_path+path[0]
                    if os.path.isdir(src_path):
                        test = test_path+path[0]+"/"
                        src_p = ""
                        for p in os.listdir(src_path):
                            p = p.split()
                            
                            src_p = src_path+"/"+p[0]
                        for i in random_numbers:
                            src = src_p+" ("+str(i)+").png"
                            print (os.path.join(test, os.path.basename(src)))
                            os.rename(src, os.path.join(test, os.path.basename(src)))

                for i in random_numbers:

                    src = src_pa+" ("+str(i)+").png"
                    print (os.path.join(test_path, os.path.basename(src)))
                    os.rename(src, os.path.join(test_path, os.path.basename(src)))
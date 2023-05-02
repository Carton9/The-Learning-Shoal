import json
import matplotlib.pyplot as plot
import numpy as np
from pathlib import Path
import os

folders=['2023-04-30T20-26-57','2023-04-30T20-21-05']
name={'2023-04-30T20-26-57':'Basic_Edit_Reward','2023-04-30T20-21-05':'LSTM_Edit_Reward'}

# path=Path("src\\tests\\scott_tests\\tiger_checkpoints") / "Basic_With_AdvReward2"/ "raw.json"
# print(path)
dataDict=None
figFolder=Path("FinalPlot")

# print(dataDict.keys())
for folder in folders:
    ylables={"moving_avg_ep_rewards":"rewards","moving_avg_ep_lengths":"steps","moving_avg_ep_avg_losses":"loss","moving_avg_ep_avg_qs":"Q value"}
    path=Path("deer_checkpoints") / folder/ "raw.json"
    
    # path=Path("src\\tests\\scott_tests\\deer_checkpoints") / folder/ "raw.json"
    # print(os.path.abspath(path))
    
    print(path)
    input()
    with open(path, "r") as f:
        dataDict=json.load(f)
    for key in list(dataDict.keys())[4:]:
        plot.cla()
        print(folder,key,len(dataDict[key]))
        # plot.plot(dataDict["ep_rewards"])
        plot.title(key.replace('_',' ')+" vs episodes")
        plot.plot(dataDict[key][:700])
        plot.xlabel("episodes")
        plot.ylabel(ylables[key])
        # if folder =="2023-04-29T16-15-21":
        #     plot.savefig(figFolder/("LSTM_With_AdvReward2"+"_tiger_"+key+'.png'))
        # else:
        # plot.savefig(figFolder/(name[folder]+"_tiger_"+key+'.png'))
        plot.savefig(figFolder/(name[folder]+"_deer_"+key+'.png'))
        plot.pause(1)
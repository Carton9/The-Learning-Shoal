import json
import matplotlib.pyplot as plot
import numpy as np
from pathlib import Path
folders=['Basic_With_AdvReward2','LSTM_With_Reward2',"Basic_With_Reward","2023-04-29T16-15-21"]

# path=Path("src\\tests\\scott_tests\\tiger_checkpoints") / "Basic_With_AdvReward2"/ "raw.json"
# print(path)
dataDict=None
figFolder=Path("src\\tests\\scott_tests\\FinalPlot")

# print(dataDict.keys())
for folder in folders:
    ylables={"moving_avg_ep_rewards":"rewards","moving_avg_ep_lengths":"steps","moving_avg_ep_avg_losses":"loss","moving_avg_ep_avg_qs":"Q value"}
    path=Path("src\\tests\\scott_tests\\tiger_checkpoints") / folder/ "raw.json"
    # path=Path("src\\tests\\scott_tests\\deer_checkpoints") / folder/ "raw.json"
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
        if folder =="2023-04-29T16-15-21":
            plot.savefig(figFolder/("LSTM_With_AdvReward2"+"_tiger_"+key+'.png'))
        else:
            plot.savefig(figFolder/(folder+"_tiger_"+key+'.png'))
        plot.pause(1)
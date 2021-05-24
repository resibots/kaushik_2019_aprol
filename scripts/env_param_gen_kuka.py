import os

env_params = {
    "controlStep": 0.01, 
    "simStep": 0.004, 
    "runtime": 2.0, 
    "jointControlMode": "position", 
    "lateral_friction": 1.0, 
    "toy": 0,
    "object_radius":0.5
}

dirs = [name for name in os.listdir(".") if os.path.isdir(name)]

for dir in dirs:
    start = dir.find("toy") + len("toy")
    end = dir.find("friction")

    data = dir[start:end].strip('_')
    toy = data.split("_")

    start = dir.find("friction") + len("friction")

    data = dir[start::].strip('_')
    friction = data.split("_")

    params = env_params

    params["toy"] = int(toy[0])
    params["lateral_friction"] = float(friction[0])

    print(params) 
    print(dir)
    print(toy)
    print(friction)
    print("---")
    
    with open(dir + '/env_params.txt', 'w') as file:
        file.write(str(env_params))
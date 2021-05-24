import os

env_params = {
    "controlStep": 0.01, 
    "simStep": 0.004, 
    "runtime": 3.0, 
    "jointControlMode": "velocity", 
    "lateral_friction": 1.0, 
    "blocked_legs": []
}

dirs = [name for name in os.listdir(".") if os.path.isdir(name)]

for dir in dirs:
	start = dir.find("block") + len("block")
	end = dir.find("friction")

	data = dir[start:end].strip('_')
	blocks = data.split("_")

	start = dir.find("friction") + len("friction")

	data = dir[start::].strip('_')
	friction = data.split("_")

	params = env_params
	if blocks[0] != "None":
		print(blocks)
		params["blocked_legs"] = [int(b) for b in blocks]

	params["lateral_friction"] = float(friction[0])

	print(params) 
 

	with open(dir + '/env_params.txt', 'w') as file:
	    file.write(str(env_params))
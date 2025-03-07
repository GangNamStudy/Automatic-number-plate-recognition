from roboflow import Roboflow
rf = Roboflow(api_key="TuqJbhofHvMRsdHE317b")
project = rf.workspace("hyunjin").project("korea-car-license-plate")
version = project.version(2)
dataset = version.download("yolov8")
#자동으로 ultralytics/cfg/datasets에 License-plate.yaml으로 넣어주는 코드 작성필요                
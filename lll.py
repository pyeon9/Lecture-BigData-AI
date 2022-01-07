import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os
import pandas as pd
import time

path = './2021-11-02_05-55-33'
ID = 2

'''
font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText = (10,30)
line_specing = 30
fontScale              = 1
fontColor1              = (255,255,255)
fontColor2              = (0,0,255)
lineType               = 2
hz = 3

# 위험행동 목록
regulation_index = np.array(['accel_lim','long_accel_lim','long_accel_sec',
                 
                 'rapid_accel_threshold','rapid_accel_acceleration',
                 'rapid_start_threshold','rapid_start_acceleration',
                 
                 'rapid_decel_threshold','rapid_decel_deceleration',
                 'rapid_stop_threshold','rapid_stop_deceleration',
                 
                 'rapid_lanechange_time',
                 'rapid_lanechange_threshold','rapid_lanechange_angle',
                 'rapid_lanechange_cangle','rapid_lanechange_defference',
                 'rapid_overtake_threshold','rapid_overtake_angle',
                 'rapid_overtake_cangle','rapid_overtake_acceleration',
                 
                 'rapid_turn_threshold','rapid_turn_sec',
                 'rapid_turn_angle_start','rapid_turn_angle_end',
                 'rapid_uturn_threshold','rapid_uturn_sec',
                 'rapid_uturn_angle_start','rapid_uturn_angle_end',
                            
                 'long_term_time', 'long_term_rest'])

# 차량 종류
regulation_columns = np.array(['truck','bus','taxi'])

# 규제 값
rehulation_values = np.array([[20,20,180,6,5,5,6,8,6,8,5,5,30,6,2,2,30,6,2,3,20,4,60,120,15,8,160,180,14400,900],
                    [20,20,180,6,6,5,8,9,6,9,5,5,30,8,2,2,30,8,2,3,25,4,60,120,20,8,160,180,14400,900],
                    [20,20,180,6,8,5,10,14,6,14,5,5,30,10,2,2,30,10,2,3,30,3,60,120,25,6,160,180,14400,900]]).T

vehicle_regulation = pd.DataFrame(rehulation_values,index=regulation_index,columns=regulation_columns )


def chagne_heading(heading):
    heading_buffer = abs(heading)*360
    if heading_buffer > 180:
        heading_buffer = 360 - heading_buffer
    return heading_buffer

class long_term_realtime:
    
    def __init__(self,vehicle):
        self.DTGdata = []
        self.vehicle = vehicle
        self.anomaly = []
        self.text = []
        #self.records = []
        
    def get_DTGdata(self,new_DTG):
        self.DTGdata.append(new_DTG)
        
    def chagne_heading(self,heading):
        heading_buffer = abs(heading)*360
        if heading_buffer > 180:
            heading_buffer = 360 - heading_buffer
        return heading_buffer
        
    def check_over_speed(self,DTGdata,last_index):
        if (DTGdata['velocity'][last_index] >= vehicle_regulation[self.vehicle]['accel_lim'] + DTGdata['road_limit'][last_index]):
            self.text.append('Speeding. Slow down.') 

            self.anomaly[-1][0] = 1
            
    def check_long_accel(self,DTGdata,last_index):
        if (DTGdata[(DTGdata.index >= last_index-vehicle_regulation[self.vehicle]['long_accel_sec']*hz) & 
            (DTGdata['velocity'] <= vehicle_regulation[self.vehicle]['accel_lim'] + DTGdata['road_limit'][last_index])].empty):
            self.text.append('Long term Speeding. Slow down.')

            self.anomaly[-1][1] = 1
                        
    def check_rapid_accel(self,DTGdata,last_index):
        if (((DTGdata['velocity'][last_index]-DTGdata['velocity'][last_index-(1*hz)]) >= vehicle_regulation[self.vehicle]['rapid_accel_acceleration']) &
            (DTGdata['velocity'][last_index] >= vehicle_regulation[self.vehicle]['rapid_accel_threshold'])):
            self.text.append('Rapid acceleration.')

            self.anomaly[-1][2] = 1
            
    def check_rapid_start(self,DTGdata,last_index):
        if (((DTGdata['velocity'][last_index]-DTGdata['velocity'][last_index-(1*hz)]) >= vehicle_regulation[self.vehicle]['rapid_start_acceleration']) &
            (DTGdata['velocity'][last_index] <= vehicle_regulation[self.vehicle]['rapid_start_threshold'])):
            self.text.append('Rapid start.')

            self.anomaly[-1][3] = 1
        
    def check_rapid_decel(self,DTGdata,last_index):
        if (((DTGdata['velocity'][last_index-1]-DTGdata['velocity'][last_index])*hz >= vehicle_regulation[self.vehicle]['rapid_decel_deceleration']) &
            (DTGdata[ 'velocity'][last_index] >= vehicle_regulation[self.vehicle]['rapid_decel_threshold'])):
            self.text.append('Rapid deceleration.')

            self.anomaly[-1][4] = 1

    def check_rapid_stop(self,DTGdata,last_index):
        if (((DTGdata['velocity'][last_index]-DTGdata['velocity'][last_index-1*hz]) >= vehicle_regulation[self.vehicle]['rapid_start_acceleration']) &
            (DTGdata['velocity'][last_index] <= vehicle_regulation[self.vehicle]['rapid_start_threshold'])):
            self.text.append('Rapid stop.')

            self.anomaly[-1][5] = 1
        
    def check_rapid_lanechange(self,DTGdata,last_index):
        if ((DTGdata['velocity'][last_index] >= vehicle_regulation[self.vehicle]['rapid_lanechange_threshold']) & 
            (self.chagne_heading(DTGdata['heading'][last_index] - DTGdata['heading'][last_index-1*hz]) >= vehicle_regulation[self.vehicle]['rapid_lanechange_angle']) & 
            (self.chagne_heading(DTGdata['heading'][last_index] - DTGdata['heading'][last_index-5*hz])/vehicle_regulation[self.vehicle]['rapid_lanechange_time'] <= vehicle_regulation[self.vehicle]['rapid_lanechange_cangle']) &
            (abs(DTGdata['velocity'][last_index] - DTGdata['velocity'][last_index-1*hz]) <= vehicle_regulation[self.vehicle]['rapid_lanechange_defference'])):
            #self.text.append('Rapid lanechange.')

            self.anomaly[-1][6] = 1
                               
    def check_rapid_overtake(self,DTGdata,last_index):
        if ((DTGdata['velocity'][last_index] >= vehicle_regulation[self.vehicle]['rapid_overtake_threshold']) & 
            (self.chagne_heading(DTGdata['heading'][last_index] - DTGdata['heading'][last_index-1*hz]) >= vehicle_regulation[self.vehicle]['rapid_overtake_angle']) & 
            (self.chagne_heading(DTGdata['heading'][last_index] - DTGdata['heading'][last_index-5*hz])/vehicle_regulation[self.vehicle]['rapid_lanechange_time'] <= vehicle_regulation[self.vehicle]['rapid_overtake_cangle']) &
            (abs(DTGdata['velocity'][last_index] - DTGdata['velocity'][last_index-1*hz])>= vehicle_regulation[self.vehicle]['rapid_overtake_acceleration'])):
            #self.text.append('Rapid overtake.')

            self.anomaly[-1][7] = 1
        
    def check_rapid_turn(self,DTGdata,last_index):
        if ((DTGdata['velocity'][last_index] >= vehicle_regulation[self.vehicle]['rapid_turn_threshold']) &
            (self.chagne_heading(DTGdata['heading'][last_index] - DTGdata['heading'][last_index-vehicle_regulation[self.vehicle]['rapid_turn_sec']*hz]) >= vehicle_regulation[self.vehicle]['rapid_turn_angle_start'])&
            (self.chagne_heading(DTGdata['heading'][last_index] - DTGdata['heading'][last_index-vehicle_regulation[self.vehicle]['rapid_turn_sec']*hz]) <= vehicle_regulation[self.vehicle]['rapid_turn_angle_end'])):
            #self.text.append('Rapid turn.')

            self.anomaly[-1][8] = 1
        
    def check_rapid_uturn(self,DTGdata,last_index):
        if ((DTGdata['velocity'][last_index] >= vehicle_regulation[self.vehicle]['rapid_uturn_threshold']) &
            (self.chagne_heading(DTGdata['heading'][last_index] - DTGdata['heading'][last_index-vehicle_regulation[self.vehicle]['rapid_uturn_sec']*hz]) >= vehicle_regulation[self.vehicle]['rapid_uturn_angle_start'])&
            (self.chagne_heading(DTGdata['heading'][last_index] - DTGdata['heading'][last_index-vehicle_regulation[self.vehicle]['rapid_uturn_sec']*hz]) <= vehicle_regulation[self.vehicle]['rapid_uturn_angle_end'])):
            #self.text.append('Rapid U turn.')

            self.anomaly[-1][9] = 1
        

        
    def check_DTGdata(self,new_DTG):
        self.text = []
        #self.records.append(np.zeros(10))
        self.get_DTGdata(new_DTG)
        DTGdata = pd.DataFrame(self.DTGdata,columns = ['time', 'velocity', 'rpm', 'brake', 'gpsX', 'gpsY', 'heading', 'accelX',
       'accelY', 'road_limit'])
        length = len(DTGdata)
        last_index = length-1
        self.anomaly.append(np.zeros(10))
        
        if ((last_index >= vehicle_regulation[self.vehicle]['rapid_uturn_sec']*hz)&
             (last_index <= vehicle_regulation[self.vehicle]['long_accel_sec']*hz)):
            self.check_over_speed(DTGdata,last_index)
            self.check_rapid_accel(DTGdata,last_index)
            self.check_rapid_start(DTGdata,last_index)
            self.check_rapid_decel(DTGdata,last_index)
            self.check_rapid_stop(DTGdata,last_index)
            self.check_rapid_lanechange(DTGdata,last_index)
            self.check_rapid_overtake(DTGdata,last_index)
            self.check_rapid_turn(DTGdata,last_index)
            self.check_rapid_uturn(DTGdata,last_index)

        elif((last_index >= vehicle_regulation[self.vehicle]['long_accel_sec']*hz)&
            (last_index <= vehicle_regulation[self.vehicle]['long_term_time']*hz)):
            self.check_over_speed(DTGdata,last_index)
            self.check_long_accel(DTGdata,last_index)
            self.check_rapid_accel(DTGdata,last_index)
            self.check_rapid_start(DTGdata,last_index)
            self.check_rapid_decel(DTGdata,last_index)
            self.check_rapid_stop(DTGdata,last_index)
            self.check_rapid_lanechange(DTGdata,last_index)
            self.check_rapid_overtake(DTGdata,last_index)
            self.check_rapid_turn(DTGdata,last_index)
            self.check_rapid_uturn(DTGdata,last_index)
            
        if(last_index >= vehicle_regulation[self.vehicle]['long_term_time']*hz):
            self.check_over_speed(DTGdata,last_index)
            self.check_long_accel(DTGdata,last_index)
            self.check_rapid_accel(DTGdata,last_index)
            self.check_rapid_start(DTGdata,last_index)
            self.check_rapid_decel(DTGdata,last_index)
            self.check_rapid_stop(DTGdata,last_index)
            self.check_rapid_lanechange(DTGdata,last_index)
            self.check_rapid_overtake(DTGdata,last_index)
            self.check_rapid_turn(DTGdata,last_index)
            self.check_rapid_uturn(DTGdata,last_index)
            self.check_long_term(DTGdata,last_index)

    def pop(self):
        if len(self.DTGdata) > 0:
            self.DTGdata.pop()
        if len(self.anomaly) > 0:
            self.anomaly.pop()
'''

def Main():
    step = 10
    go = True
    k = 0
    target_danger = 0
    wait = 100
    # truck_realtime = long_term_realtime('truck')

    try:
        os.makedirs(f'{path}/danger/')
    except:
        pass
    while go:
        delta = 1
        frames = 1
        if frames == '0':
            break
        try: 
            step = int(frames)
        except:
            pass


        if k >= len(os.listdir(path+f'/driver')):
            break
        driver = cv2.imread(path+f'/driver/{k}.jpg')
        cv2.imshow('driver', driver)

        screen = cv2.imread(path+f'/screen_shots/{k}.jpg')
        screen = cv2.resize(screen, (1600,900))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        cv2.imshow('screen', screen)

        log_screen = np.zeros((200, 900, 3))

        # with open(path+f'/log/{k}.json','r') as f:
        #     drive_data = json.load(f)
        # val = [k, drive_data['truck']['speed'],drive_data['truck']['engineRpm'],
        #                   drive_data['truck']['userBrake'],drive_data['truck']['placement']['x'],
        #                   drive_data['truck']['placement']['y'],
        #                   drive_data['truck']['userSteer'],drive_data['truck']['acceleration']['x'],
        #                   drive_data['truck']['acceleration']['y'],drive_data['navigation']['speedLimit']]
        # truck_realtime.check_DTGdata(val)
        # log = f'idx: {k} spd: {val[1]:.2f}, lim: {val[-1]}'
        # if len(truck_realtime.text) > 0:
        #     violation = True
        # else:
        #     violation = False
        # if violation:
        #     fontColor = fontColor2
        # else:
        #     fontColor = fontColor1

        # cv2.putText(log_screen, log,
        #         (10, 30),
        #         font,
        #         fontScale,
        #         fontColor,
        #         lineType)
        # for n, txt in enumerate(truck_realtime.text):
        #     cv2.putText(log_screen, txt,
        #         (10, 30*(2+n)),
        #         font,
        #         fontScale,
        #         fontColor,
        #         lineType)

        cv2.imshow('log', log_screen)

        key = cv2.waitKey(wait)

        kdict = {'num0': 48, 'num1': 49, 'num2': 50, 'num3': 51, 'num4': 52, 'num5': 53, 'num6': 54,
                 'space': 32, 'a': 97, 's': 115, 'd': 100, 'f': 102, 'esc': 27, 'back': 8}
        if key == kdict['esc']:
            go=False
        elif key == kdict['num0']:
            target_danger = 0
        elif key == kdict['num1']:
            target_danger = 1
        elif key == kdict['num2']:
            target_danger = 2
        elif key == kdict['num3']:
            target_danger = 3
        elif key == kdict['num4']:
            target_danger = 4
        elif key == kdict['num5']:
            target_danger = 5
        elif key == kdict['num5']:
            target_danger = 6

        elif key == kdict['a']:
            wait = 1000
        elif key == kdict['s']:
            wait = 500
        elif key == kdict['d']:
            wait = 100
        elif key == kdict['f']:
            wait = 50

        elif key == kdict['space']:
            wait = 0

        elif key ==  kdict['back']:
            delta = -1
            wait = 0
            # truck_realtime.pop()
            # truck_realtime.pop()

        elif key == -1:
            pass
        else:
            print(key)



        with open(f'{path}/danger/label_{k}','w') as f:
            rule_based = 0
            if truck_realtime.anomaly[-1][7] == 1:
                rule_based += 1

            if truck_realtime.anomaly[-1][8] == 1:
                rule_based += 1

            if truck_realtime.anomaly[-1][9] == 1:
                rule_based += 1

            if truck_realtime.anomaly[-1][6] == 1:
                rule_based += 1
            f.write(f"{target_danger + rule_based}\n")
            print((f"{k}, {target_danger:.2f}\n"))

        k += delta


if __name__ == '__main__':
    Main()
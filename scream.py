


import subprocess
process = subprocess.Popen(['python3', 'droning.py'], stdout=subprocess.PIPE)
stdout = process.communicate()[0]
print ('STDOUT:{}'.format(stdout))
# conected=True
# if (conected):

#     while True:
#         label=str(input()).strip()

#         f=open("1s.txt", 'a')
#         f.write(label)
#         f.close
        
#         if label=="okay":
#             f=open("okay.txt", 'a')
#             f.write(label)
#             f.close
#         elif label=="C": 
#             #print('forward')
#             #bebop.fly_direct(0, 10, 0, 0, 0.1)
#             f=open("c.txt", 'a')
#             f.write(label)
#             f.close
#         elif label=="L": 
#             # print('backward')
#             # bebop.fly_direct(0, -10, 0, 0, 0.1)
#             f=open("l.txt", 'a')
#             f.write(label)
#             f.close
#         elif label=="fist": 
#             # print('left')
#             # bebop.fly_direct(-10, 0, 0, 0, 0.1)
#             f=open("fist.txt", 'a')
#             f.write(label)
#             f.close
#         # if label=="okay": 
#         #     print('right')
#         #     bebop.fly_direct(10, 0, 0, 0, 0.1)
#         elif label=="palm": 
#             # print('up')
#             # bebop.fly_direct(0, 0, 0,10, 0.1)
#             f=open("palm.txt", 'a')
#             f.write(label)
#             f.close
#         elif label=="peace": 
#             # print('up')
#             # bebop.fly_direct(0, 0, 0,10, 0.1)
#             f=open("peace.txt", 'a')
#             f.write(label)
#             f.close
#         # if keyboard.is_pressed('Esc'): 
#         #     break

#     # print("DONE - DISCONNECTING")
#     # print("Battery is %s" % bebop.sensors.battery)  #imprime el estado de la bateria
#     # # bebop.disconnect()
#     # # cam_capture.release()
#     # # cv2.destroyAllWindows()

#     # echo "peace" > /proc/23005/fd/0
        
        
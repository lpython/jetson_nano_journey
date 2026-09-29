```bash
(base) jetson@nano:/sys/devices/pwm-fan$ ls
cur_pwm                 fan_kickstart     kickstart_params  pwm_cap        state_cap    target_pwm
driver                  fan_profile       modalias          pwm_rpm_table  step_time    temp_control
driver_override         fan_rpm_in_limit  of_node           pwm_state_map  subsystem    uevent
fan_available_profiles  hwmon             power             rpm_measured   tach_enable
(base) jetson@nano:/sys/devices/pwm-fan$ cat cur_pwm
255
(base) jetson@nano:/sys/devices/pwm-fan$ cat target_pwm
255
(base) jetson@nano:/sys/devices/pwm-fan$ cat fan_profile 
N/A
(base) jetson@nano:/sys/devices/pwm-fan$ cat pwm_rpm_table 
(Index, RPM, PWM, RRU, RRD)
(0, 0, 0, 40, 40)
(1, 1000, 80, 2, 2)
(2, 2000, 120, 1, 1)
(3, 3000, 160, 1, 1)
(4, 4000, 255, 1, 1)
(5, 5000, 255, 1, 1)
(6, 6000, 255, 1, 1)
(7, 7000, 255, 1, 1)
(8, 10000, 255, 1, 1)
(9, 11000, 255, 1, 1) 
(base) jetson@nano:/sys/devices/pwm-fan$ sudo nvpmodel -q
NVPM WARN: fan mode is not set!
NV Power Mode: MAXN
0
(base) jetson@nano:/sys/devices/pwm-fan$ sudo nvpmodel -p quiet
```
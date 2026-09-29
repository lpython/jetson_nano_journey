

### Setup VNC

```bash

sudo apt search x11vnc
sudo apt-get install x11vnc

# change passwd file permission, match service.user
sudo x11vnc -storepasswd /etc/x11vnc.pass
sudo chown jetson:jetson /etc/x11vnc.pass
sudo chmod 600 /etc/x11vnc.pass

sudo cp x11vnc.service  /etc/systemd/system/x11vnc.service

sudo systemctl deamon-reload
sudo systemctl enable x11vnc.service
sudo systemctl restart x11vnc.service
sudo systemctl status x11vnc.service

```


x11vnc.service
```
[Unit] 
Description=x11vnc VNC Server
Requires=display-manager.service
After=display-manager.service

[Service]
Type=simple
User=jetson
ExecStart=/usr/bin/x11vnc -auth guess -display WAIT:0 -forever -loop -noxdamage -repeat -rfbauth /etc/x11vnc.pass -rfbport 5990 -shared
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
(base) jetson@nano:~$ 
```
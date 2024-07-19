import socket
import pygame
import threading 
import time

locker = threading.Lock()

pygame.mixer.init()

class client_thread(threading.Thread):
    def __init__(self, client_address = None, client_socket = None) -> None:
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.alert = ""
        print("Connected to client_address: " , client_address)
    def run(self) -> None:
        while True:
            try:
                data = self.client_socket.recv(1024)
            except Exception as e:
                self.client_socket.close()
                break

            if(not data):
                self.client_socket.close()
                break 

            alert = data.decode()
            print(alert)
            
            locker.acquire()
            if(self.alert != alert):
                self.alert = alert
                pygame.mixer.music.load(alert + ".mp3")
            pygame.mixer.music.play()
            locker.release()
            #self.client_socket.send("ACK!".encode())
            time.sleep(3)
            self.client_socket.recv(1024)

bind_ip = "127.0.0.1"
bind_port = 9999

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

server.bind((bind_ip,bind_port))

server.listen(5)

print("server listen on " )
print((bind_ip,bind_port))

while True:
    client,addr = server.accept()
    newthread = client_thread(addr, client)
    newthread.start()
    
        
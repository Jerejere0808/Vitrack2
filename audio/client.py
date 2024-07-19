import socket
from time import sleep
HOST = '127.0.0.1'
PORT = 9999

class client:

    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
         
    def set_connection(self , connection):
        self.connection = connection
        try:
            self.s.connect(self.connection)
        except Exception as e:
            print(e)

    def send_alert(self, alert):
        self.s.send(alert.encode())
        #data = self.s.recv(1024)
        #print(data)
        
    def close(self):
        self.s.close()
        
#You can test here
if __name__ == '__main__':
    cli = client()
    cli.set_connection((HOST , PORT))
    i = 0
    while(i < 2):
        cli.send_alert("stay right")
        i+=1
        sleep(3)

    cli.close()
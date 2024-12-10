from locust import HttpUser, task
from main import app
from starlette.testclient import TestClient

# client = TestClient(app)

class HelloWorldUser(HttpUser):
    @task
    def cellphone_test(self):
        files = {
            'file': ('1.jpg', open(r'C:\Users\Sepehr\Desktop\project1\Images\5\4.jpg', 'rb'), 'image/jpeg'),
        }
        response = self.client.post("/cellphone", files=files)
        assert response.status_code == 200
    @task
    def seatbelt_test(self):
        files = {
            'file':('1.jpg', open(r'C:\Users\Sepehr\Desktop\project1\Images\4\9.jpg', 'rb'), 'image/jpeg')
        }
        response = self.client.post('/seatbelt', files=files)
        assert response.status_code==200

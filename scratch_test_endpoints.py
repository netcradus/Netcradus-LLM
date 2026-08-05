import os
import sys
import time
import subprocess
import requests

# Start the web server on port 8099
print("Starting web server...")
server_proc = subprocess.Popen(
    [sys.executable, "web_server.py", "--port", "8099"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(3) # Wait for startup

base_url = "http://localhost:8099/api"

try:
    # 1. Register
    print("Testing Registration...")
    reg_data = {
        "username": "api_test_user",
        "password": "password123",
        "display_name": "API Test User"
    }
    r = requests.post(f"{base_url}/user/auth/register", json=reg_data)
    print("Reg response:", r.status_code, r.json())

    # 2. Login
    print("\nTesting Login...")
    login_data = {
        "username": "api_test_user",
        "password": "password123"
    }
    r = requests.post(f"{base_url}/user/auth/login", json=login_data)
    print("Login response:", r.status_code, r.json())
    token = r.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Get Profile
    print("\nTesting Get Profile...")
    r = requests.get(f"{base_url}/user/profile", headers=headers)
    print("Get Profile response:", r.status_code, r.json())

    # 4. Update Profile
    print("\nTesting Update Profile...")
    update_data = {
        "display_name": "Updated API User",
        "bio": "Developer testing APIs",
        "avatar_color": "#ffaa00"
    }
    r = requests.put(f"{base_url}/user/profile", headers=headers, json=update_data)
    print("Update Profile response:", r.status_code, r.json())

    # 5. Change Password
    print("\nTesting Change Password...")
    pass_data = {
        "current_password": "password123",
        "new_password": "newpassword123"
    }
    r = requests.post(f"{base_url}/user/change-password", headers=headers, json=pass_data)
    print("Change Password response:", r.status_code, r.json())

except Exception as e:
    print("Exception during tests:", e)

finally:
    print("\nShutting down web server...")
    server_proc.terminate()
    server_proc.wait()
    print("Done.")

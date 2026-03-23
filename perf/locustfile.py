from locust import HttpUser, task, between


class StreamUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def chat_stream(self):
        payload = {
            "prompt": "Give me a short update on crypto market movement today",
            "session_id": f"locust-{self.environment.runner.user_count}-{id(self)}",
        }
        with self.client.post(
            "/chat/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=65,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200 or "[DONE]" not in resp.text:
                resp.failure("stream response invalid")
            else:
                resp.success()

    @task(1)
    def chat_non_stream(self):
        payload = {
            "prompt": "What is Ethereum price right now?",
            "session_id": f"locust-chat-{id(self)}",
        }
        self.client.post("/chat", json=payload, timeout=30)

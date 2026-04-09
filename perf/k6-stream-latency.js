import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  scenarios: {
    steady_stream: {
      executor: "ramping-arrival-rate",
      startRate: 5,
      timeUnit: "1s",
      preAllocatedVUs: 20,
      maxVUs: 200,
      stages: [
        { target: 20, duration: "1m" },
        { target: 40, duration: "2m" },
        { target: 40, duration: "2m" },
        { target: 0, duration: "30s" },
      ],
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<3000", "p(99)<6000"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";

export default function () {
  const payload = JSON.stringify({
    prompt: "what is the latest bitcoin price and major market update",
    session_id: `k6-${__VU}-${__ITER}`,
  });

  const params = {
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    timeout: "65s",
  };

  const res = http.post(`${BASE_URL}/chat/stream`, payload, params);
  check(res, {
    "status is 200": (r) => r.status === 200,
    "stream ended": (r) => r.body && r.body.includes("[DONE]"),
  });

  sleep(0.2);
}

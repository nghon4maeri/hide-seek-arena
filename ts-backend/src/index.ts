import { createServer } from "./server.js";

const PORT = Number(process.env.PORT) || 3001;

createServer().listen(PORT, () => {
  console.log(`[ts-backend] Arena server running at http://localhost:${PORT}`);
  console.log(`[ts-backend] SSE endpoint: http://localhost:${PORT}/api/sse`);
  console.log(`[ts-backend] Health: http://localhost:${PORT}/api/health`);
});

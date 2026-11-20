declare module 'ws' {
  import type { EventEmitter } from 'node:events';
  class WebSocket extends EventEmitter {
    constructor(address: string);
    send(data: string): void;
    close(): void;
  }
  export default WebSocket;
}

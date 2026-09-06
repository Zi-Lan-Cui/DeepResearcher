# HTTPS 部署

## 结论

生产环境必须使用证书，但应用不应自己管理证书。推荐让 Caddy 监听公网
`80/443`，自动申请、续期和加载证书；FastAPI/Uvicorn 仍只监听本机
`127.0.0.1:8080`。这样 TLS 生命周期与 Python 进程解耦，API 和 Worker 的代码
都不需要因为证书轮换而重启或修改。

```text
浏览器
  │ HTTPS / SSE
  ▼
Caddy :443                 自动证书、HTTP→HTTPS、响应头
  │ HTTP（仅本机回环）
  ▼
Uvicorn 127.0.0.1:8080     FastAPI API + 静态前端
  │
  ├── PostgreSQL
  └── Redis（可丢弃的流式预览）
```

## 公网域名：无需手工购买证书

准备条件：

1. 拥有一个域名，例如 `research.example.com`。
2. 域名的 A/AAAA 记录指向服务器公网地址。
3. 安全组和防火墙允许外部访问 TCP 80、443。
4. 8080 不对公网开放；`SERVICE_HOST=127.0.0.1`。
5. Caddy 对证书存储目录具有持久写权限。

先启动 API：

```bash
uv run python server.py
```

再启动 Caddy：

```bash
DEEPSEARCH_SITE_ADDRESS=research.example.com \
  caddy run --config deploy/Caddyfile
```

Caddy 会自动完成 ACME 证书申请、HTTP 到 HTTPS 跳转和续期。页面、API、SSE
全部使用同一 HTTPS 源，因此前端代码不需要硬编码协议或域名。

生产配置保持：

```dotenv
APP_ENV=production
SERVICE_HOST=127.0.0.1
SERVICE_PORT=8080
SERVICE_FORWARDED_ALLOW_IPS=127.0.0.1
```

`SERVICE_FORWARDED_ALLOW_IPS` 表示 Uvicorn 只接受本机 Caddy 提供的
`X-Forwarded-For` / `X-Forwarded-Proto`。登录 IP 限流依赖这里得到真实客户端
地址。不要在 Uvicorn 直接暴露公网时将其设置成 `*`，否则客户端可以伪造来源 IP
绕过限流。

## 本地与内网

默认 `DEEPSEARCH_SITE_ADDRESS=localhost` 时，Caddy 使用本地 CA 签发开发证书。
浏览器要完全信任它，需要把 Caddy 本地根证书加入操作系统信任库。若只做日常开发，
继续访问 `http://127.0.0.1:8080` 更简单；HTTPS 是公网交付边界，不要求每次本地调试
都启用。

没有公共 DNS、80/443 无法入站或服务器离线时，公共 CA 无法完成验证。这时可以：

- 使用 `mkcert` 为本机/内网名称生成受本机信任的开发证书；
- 使用企业内部 CA；
- 在 Caddyfile 中配置已有的证书和私钥。

自签名证书只能提供加密，默认不会被其他用户的浏览器信任，不适合作为公网正式证书。

## 验收

```bash
curl -I http://research.example.com
curl -I https://research.example.com
curl -N https://research.example.com/api/runs/RUN_ID/events \
  -H 'Authorization: Bearer TOKEN'
```

验收标准：HTTP 返回到 HTTPS 的跳转；HTTPS 证书链可信且域名匹配；页面、API 和
长连接 SSE 均正常；公网不能直接连接 8080。

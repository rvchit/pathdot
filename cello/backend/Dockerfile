# Backend: Express + TypeScript, compiled with tsc, run with node.
# glibc image (not alpine) so bcrypt/sharp use prebuilt binaries.
FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY tsconfig.json ./
COPY src ./src
RUN npm run build && npm prune --omit=dev

FROM node:22-bookworm-slim
WORKDIR /app
ENV NODE_ENV=production \
    PORT=4001 \
    STORAGE_MODE=local \
    TILES_DIR=/srv/tiles \
    DATA_DIR=/srv/data \
    UPLOAD_DIR=/srv/data/slides \
    ML_SERVICE_URL=http://ml:8008
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY package.json ./
COPY prisma ./prisma
COPY scripts ./scripts
RUN mkdir -p /srv/tiles /srv/data/slides
EXPOSE 4001
HEALTHCHECK --interval=30s --timeout=3s CMD node -e "fetch('http://localhost:4001/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "dist/server/index.js"]

const internalApiUrl = process.env.NEXT_INTERNAL_API_URL ?? "http://backend:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${internalApiUrl}/api/:path*`,
      },
      {
        source: "/static/:path*",
        destination: `${internalApiUrl}/static/:path*`,
      },
    ];
  },
};

export default nextConfig;

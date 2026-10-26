// src/config.js
/**
 * Environment Configuration for DEWANTARA
 *
 * Endpoints follow the host: a local dev server talks to the local API, anything
 * else to production. Both can be overridden at build time with
 * REACT_APP_API_BASE_URL and REACT_APP_WS_URL (CRA inlines these at build time).
 */

const apiBaseUrlOverride = process.env.REACT_APP_API_BASE_URL;
const wsUrlOverride = process.env.REACT_APP_WS_URL;

const isDevelopment =
	window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
const isProduction = window.location.protocol === "https:";

// Strip trailing slashes so a configured base URL can be joined safely.
const API_BASE_URL = (apiBaseUrlOverride || (isDevelopment
	? "http://localhost:8000"
	: "https://api.dewantara.cloud")).replace(/\/+$/, "");

// http:// -> ws:// and https:// -> wss://
const WS_URL = wsUrlOverride || `${API_BASE_URL.replace(/^http/, "ws")}/ws`;

// Configuration object
const config = {
	// API Base URLs
	API_BASE_URL,

	// WebSocket URL
	WS_URL,

	// Environment flags
	isDevelopment,
	isProduction,

	// App Info
	APP_NAME: "DEWANTARA",
	APP_VERSION: "2.0.0",

	// Feature flags
	ENABLE_LOGGING: isDevelopment,
	ENABLE_ANALYTICS: isProduction,
};

export default config;

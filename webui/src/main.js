// Main entry point for the HASS-MCP+ Web UI
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import './assets/styles/main.css'

// Load environment variables
const env = window.env || {
  API_URL: 'http://localhost:8787',
  VERSION: '1.0.0',
  ENV: 'development'
}

// Create app instance
const app = createApp(App)

// Configure global properties
app.config.globalProperties.$env = env
app.config.globalProperties.$api = {
  baseUrl: env.API_URL,
  async query(text) {
    const response = await fetch(`${env.API_URL}/api/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({ query: text })
    })
    return await response.json()
  },
  // Additional API methods would be defined here
}

// Mount the app
app
  .use(store)
  .use(router)
  .mount('#app')

console.log(`HASS-MCP+ Web UI ${env.VERSION} started in ${env.ENV} mode`)
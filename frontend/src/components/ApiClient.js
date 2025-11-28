import axios from 'axios';
import Constants from 'expo-constants';

const API_BASE_URL = (Constants?.manifest?.extra?.AEVORIUM_API_URL) || 'http://localhost:8000';

let token = null;

export const setApiToken = (t) => { token = t };

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

api.interceptors.request.use(config => {
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;

import axios from 'axios';

// Hardcoded API URL - update this to match your server's IP
const API_BASE_URL = 'http://10.175.165.124:8000';

let token = null;

export const setApiToken = (t) => { token = t; };
export const getApiToken = () => token;
export const getApiBaseUrl = () => API_BASE_URL;

// Create axios instance with minimal config to avoid URL parsing issues
export const api = axios.create({
  timeout: 120000,
});

// Set baseURL after creation to avoid URL constructor issues
api.defaults.baseURL = API_BASE_URL;
api.defaults.headers.common['Content-Type'] = 'application/json';

// Request interceptor - add auth token
api.interceptors.request.use(
  config => {
    if (token) {
      config.headers = config.headers || {};
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// Response interceptor - handle errors globally
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.status, error.response.data);
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error: No response received');
    } else {
      // Error setting up request
      console.error('Request Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// API methods for convenience
export const apiMethods = {
  // Health check
  healthCheck: () => api.get('/health'),
  
  // Privacy budget
  getPrivacyBudget: () => api.get('/privacy-budget'),
  updatePrivacyBudget: (data) => api.post('/privacy-budget', data),
  
  // Generate synthetic data
  generateData: (params) => api.post('/generate', params),
  
  // Training
  startTraining: (params) => api.post('/train', params),
  getTrainingStatus: () => api.get('/training-status'),
  
  // Audit log
  getAuditLog: () => api.get('/audit-log'),
  clearHistory: () => api.delete('/clear-history'),
  
  // Dataset
  getDataset: (params) => api.get('/dataset', { params }),
  getDatasetStats: () => api.get('/dataset/stats'),
  downloadDataset: () => api.get('/dataset/download', { responseType: 'blob' }),
  getDatasetSample: (n = 10) => api.get(`/dataset/sample?n=${n}`),
};

export default api;

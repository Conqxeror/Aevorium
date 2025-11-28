# Aevorium Mobile App (Expo React Native)

This is a minimal Expo-based React Native app that mirrors the dashboard features so users can monitor and interact with the Aevorium backend on mobile devices.

Features:
- Overview (API health, privacy spend, recent audit events)
- Generate synthetic data via API (with client-side validation)
- Inspect privacy budget and set/reset it (if authorized)
- View audit log entries

Requirements:
- Node.js 18+ and npm, or Yarn
- Expo CLI (optional: npx expo)
- Running Aevorium backend (API) at `AEVORIUM_API_URL` in `app.json` or set on `app.config.js` extras

Run the app:

```powershell
cd frontend
npm install
npm start
# then follow Expo UI to run on device or emulator
```

Important notes:
- The app uses an API token header (`Authorization: Bearer <token>`) if provided in the app settings.
- The `Generate` endpoint respects server-side limits; client validates sample count and filename as a safety measure.

Security:
- Do not commit tokens in source or `app.json`.
- Use the API token for auth when deploying in protected environments.

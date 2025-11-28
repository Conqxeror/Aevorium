import 'dotenv/config';

export default {
  expo: {
    name: 'Aevorium Mobile',
    slug: 'aevorium-mobile',
    extra: {
      AEVORIUM_API_URL: process.env.AEVORIUM_API_URL || 'http://localhost:8000'
    },
  },
};

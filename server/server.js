const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');
require('dotenv').config();

const chatRoutes = require('./routes/chatRoutes'); // ✅ Ensure correct path

const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Use Routes
app.use('/api', chatRoutes); // ✅ Correct usage

// Database Connection
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => console.log('MongoDB connected'))
  .catch(err => console.error(err));

// Start Server
const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

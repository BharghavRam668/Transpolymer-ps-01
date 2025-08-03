import express from 'express';
import ResetRequest from '../model/resetRequestSchema.js';
import User from '../model/userSchema.js';

const router = express.Router();

// POST: Submit a password reset request
router.post('/', async (req, res) => {
  const { email, desiredPassword } = req.body;

  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return res.status(400).json({ message: 'Invalid email address.' });
  }

  if (!desiredPassword || desiredPassword.length < 8) {
    return res.status(400).json({ message: 'Desired password must be at least 8 characters long.' });
  }

  try {
    const userExists = await User.exists({ email });
    if (!userExists) {
      return res.status(404).json({ message: 'No user found with this email.' });
    }

    await ResetRequest.create({ email, desiredPassword });
    res.status(200).json({ message: 'Reset request submitted' });
  } catch (err) {
    console.error('Reset request error:', err);
    res.status(500).json({ message: 'Error processing reset request' });
  }
});

// GET: For admins to see reset requests
router.get('/admin', async (req, res) => {
  try {
    const requests = await ResetRequest.find().sort({ createdAt: -1 });
    res.json(requests);
  } catch (err) {
    res.status(500).json({ message: 'Error retrieving requests' });
  }
});

export default router;

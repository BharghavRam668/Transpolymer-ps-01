import express from 'express';
import { registerUser, loginUser, getUserProfile } from '../controller/userController.js';
import User from '../model/userSchema.js';

const router = express.Router();

// ✅ Register and login routes
router.post('/register', registerUser);
router.post('/login', loginUser);

// ✅ Get regular users (for admin dashboard)
router.get('/regular-users', async (req, res) => {
  try {
    const users = await User.find({}, 'email username');
    res.json(users);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching users' });
  }
});

// (Optional) Protected route example
// router.get('/profile', userMiddleware, getUserProfile);

export default router;

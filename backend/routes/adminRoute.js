import express from 'express';
import Admin from '../model/adminSchema.js';
import ResetRequest from '../model/resetRequestSchema.js';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

const router = express.Router();

// Admin login
router.post('/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    const admin = await Admin.findOne({ email });
    if (!admin) return res.status(401).json({ message: 'Invalid email or password' });

    const isMatch = await admin.comparePassword(password);
    if (!isMatch) return res.status(401).json({ message: 'Invalid email or password' });

    const token = jwt.sign({ id: admin._id, email: admin.email }, process.env.JWT_SECRET, { expiresIn: '1h' });

    res.json({ token, admin: { email: admin.email, id: admin._id } });
  } catch (error) {
    console.error('Admin login error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Return admin list
router.get('/admin-users', async (req, res) => {
  try {
    const admins = await Admin.find({}, 'email'); // Return email only
    res.json(admins);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching admins' });
  }
});

// Approve reset request & update user password (auto use desiredPassword from request)
router.post('/approve-reset/:id', async (req, res) => {
  try {
    const resetRequest = await ResetRequest.findById(req.params.id);
    if (!resetRequest) return res.status(404).json({ message: 'Reset request not found' });

    const userEmail = resetRequest.email;
    const desiredPassword = resetRequest.desiredPassword;
    if (!desiredPassword) return res.status(400).json({ message: 'Desired password not found in request' });

    const User = (await import('../model/userSchema.js')).default;
    const user = await User.findOne({ email: userEmail });
    if (!user) return res.status(404).json({ message: 'User not found' });

    // Hash the desiredPassword before saving
    const hashedPassword = await bcrypt.hash(desiredPassword, 10);
    user.password = hashedPassword;
    await user.save();

    // Delete the reset request after password update
    await ResetRequest.findByIdAndDelete(req.params.id);

    // Respond with message suitable for popup notification
    res.json({ success: true, message: 'Password has been changed.' });
  } catch (error) {
    console.error('Error approving reset:', error);
    res.status(500).json({ message: 'Error approving reset' });
  }
});

// Deny reset request (deletes the request)
router.post('/deny-reset/:id', async (req, res) => {
  try {
    await ResetRequest.findByIdAndDelete(req.params.id);
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ message: 'Error denying reset' });
  }
});

export default router;

import jwt from 'jsonwebtoken';
import Admin from '../model/adminSchema.js';  // Your admin model with email/password or roles

export const verifyAdminToken = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ message: 'Unauthorized: No token provided' });
    }

    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET);

    // Verify user is admin
    const adminUser = await Admin.findById(decoded.id);
    if (!adminUser) {
      return res.status(403).json({ message: 'Forbidden: Admin access only' });
    }

    req.admin = adminUser;  // Attach admin info to request if needed
    next();
  } catch (err) {
    return res.status(401).json({ message: 'Unauthorized: Invalid token' });
  }
};

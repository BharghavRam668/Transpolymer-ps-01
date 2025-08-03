import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import cookieParser from "cookie-parser";
import dotenv from "dotenv";

import userRoutes from "./routes/userRoute.js";
import searchHistoryRoute from "./routes/searchHistoryRoute.js";
import resetRequestRoute from "./routes/ResetRequestRoute.js";  // Correct casing if needed
import adminRoutes from "./routes/adminRoute.js";  // Admin login/logout and admin routes
import connectDB from "./db/connect.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

connectDB();

app.use(cookieParser());
app.use(bodyParser.json());

app.use(
  cors({
    origin: ["http://localhost:3000", "http://localhost:3001"], // Add all frontend origins here
    credentials: true,
  })
);

app.use("/api/users", userRoutes);
app.use("/api/search-history", searchHistoryRoute);
app.use("/api/reset-request", resetRequestRoute);
app.use("/api/admin", adminRoutes);

app.get("/", (req, res) => res.send("Welcome to Transpolymer API"));

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});

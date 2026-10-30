import { Router } from "express";
import bcrypt from "bcrypt";

const router = Router();

// Mock user database (replace with a real database)
const users: { username: string; password: string }[] = [];

// Register route
router.post("/register", async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ message: "Username and password are required" });
  }

  // Check if the user already exists
  const existingUser = users.find((user) => user.username === username);
  if (existingUser) {
    return res.status(400).json({ message: "User already exists" });
  }

  // Hash the password
  const hashedPassword = await bcrypt.hash(password, 10);

  // Store the user
  users.push({ username, password: hashedPassword });

  console.log("Registered Users:", users);

  res.status(201).json({ message: "User created successfully" });


});

// Login route
router.post("/login", async (req, res) => {
    const { username, password } = req.body;
  
    if (!username || !password) {
      return res.status(400).json({ message: "Username and password are required" });
    }
  
    // Find the user by username
    const user = users.find((u) => u.username === username);
    if (!user) {
      return res.status(401).json({ message: "Invalid credentials" });
    }
  
    // Compare the provided password with the stored hashed password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ message: "Invalid credentials" });
    }
  
    console.log("Login successful for user:", username);
    res.status(200).json({ message: "Login successful" });
  });

export default router;
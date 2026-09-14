"""
Context examples for project estimation.
These examples are used as references for the LLM to generate new estimations.
"""

ESTIMATION_EXAMPLES = [
    {
        "project_name": "Hyrox Workout Tracker App",
        "meeting_summary": """
Meeting Date: 2025-09-10
Participants: Product Owner, Engineering Lead, 2 Developers

Objective: Build a mobile/web application to track Hyrox workout performance metrics.
Hyrox is an 8km obstacle course fitness competition that includes running segments and 8 workout stations.

Key Requirements Discussed:
- Users can log completed Hyrox workouts with detailed metrics
- Real-time tracking of heart rate zones during workout
- Calorie burn calculation and analytics
- Performance history and trend analysis
- Export workout data to PDF/CSV

User Stories Refined:
1. User Authentication & Profile Management
2. Workout Data Entry & Real-time Logging
3. Heart Rate Zone Tracking & Analysis
4. Calorie Burn Calculation Engine
5. Dashboard with Analytics & Visualizations
6. Data Export Functionality
7. Mobile Responsiveness
8. Performance Optimization & Testing
        """,
        "tasks": [
            {
                "task_id": 1,
                "name": "User Authentication & Profile Setup",
                "description": "Implement user registration, login, and profile management with email verification",
                "estimated_hours": 16,
                "estimated_cost_usd": 640,
                "complexity": "Medium",
                "includes": [
                    "User registration endpoint",
                    "Email verification",
                    "JWT token management",
                    "Profile CRUD operations",
                    "Password reset flow"
                ]
            },
            {
                "task_id": 2,
                "name": "Workout Data Entry Form",
                "description": "Create UI form to capture workout details including distance, time, calories, heart rate zones",
                "estimated_hours": 20,
                "estimated_cost_usd": 800,
                "complexity": "Medium",
                "includes": [
                    "Form validation",
                    "Date/time pickers",
                    "Zone input fields",
                    "Unit conversion (km/miles, kcal/kJ)",
                    "Form error handling"
                ]
            },
            {
                "task_id": 3,
                "name": "Heart Rate Zone Tracking System",
                "description": "Implement algorithm to categorize heart rate zones and calculate time spent in each zone",
                "estimated_hours": 24,
                "estimated_cost_usd": 960,
                "complexity": "High",
                "includes": [
                    "Zone calculation engine",
                    "Real-time data processing",
                    "Integration with HR devices/APIs",
                    "Zone threshold configuration",
                    "Historical zone data storage"
                ]
            },
            {
                "task_id": 4,
                "name": "Calorie Burn Calculator",
                "description": "Develop algorithm to calculate calories burned based on activity intensity, duration, and user metrics",
                "estimated_hours": 18,
                "estimated_cost_usd": 720,
                "complexity": "High",
                "includes": [
                    "Calorie calculation algorithm",
                    "User profile factors (age, weight, gender)",
                    "Activity intensity factors",
                    "Validation against industry standards",
                    "Unit conversion support"
                ]
            },
            {
                "task_id": 5,
                "name": "Analytics Dashboard Development",
                "description": "Create dashboard with charts, graphs, and performance metrics visualization",
                "estimated_hours": 32,
                "estimated_cost_usd": 1280,
                "complexity": "High",
                "includes": [
                    "Chart implementation (Chart.js/Recharts)",
                    "Performance trends visualization",
                    "Zone distribution pie charts",
                    "Calorie vs time graphs",
                    "Comparison views (workout vs workout)",
                    "Export button integration"
                ]
            },
            {
                "task_id": 6,
                "name": "Data Export Functionality",
                "description": "Implement PDF and CSV export capabilities for workout data and reports",
                "estimated_hours": 14,
                "estimated_cost_usd": 560,
                "complexity": "Medium",
                "includes": [
                    "PDF generation (report format)",
                    "CSV export with formatting",
                    "Data filtering for export",
                    "Template design",
                    "Error handling for large datasets"
                ]
            },
            {
                "task_id": 7,
                "name": "Mobile Responsiveness & UI Polish",
                "description": "Ensure application is fully responsive and optimized for mobile devices",
                "estimated_hours": 16,
                "estimated_cost_usd": 640,
                "complexity": "Medium",
                "includes": [
                    "Responsive design implementation",
                    "Touch-friendly interfaces",
                    "Mobile-specific optimizations",
                    "Performance optimization",
                    "Cross-browser testing"
                ]
            },
            {
                "task_id": 8,
                "name": "Testing & Quality Assurance",
                "description": "Comprehensive testing including unit tests, integration tests, and end-to-end testing",
                "estimated_hours": 28,
                "estimated_cost_usd": 1120,
                "complexity": "Medium",
                "includes": [
                    "Unit test coverage",
                    "Integration tests",
                    "E2E test automation",
                    "Performance testing",
                    "Security testing",
                    "Bug fixes"
                ]
            }
        ],
        "summary": {
            "total_hours": 168,
            "total_cost_usd": 6720,
            "team_size": "2 developers",
            "estimated_duration_weeks": 5,
            "hourly_rate": 40
        }
    },
    {
        "project_name": "E-Commerce Platform Redesign",
        "meeting_summary": """
Meeting Date: 2025-09-12
Participants: Product Manager, UX Designer, Tech Lead, 3 Backend Devs, 2 Frontend Devs

Objective: Complete redesign and modernization of existing e-commerce platform.
Current system built on legacy monolith, needs migration to microservices with new UI.

Key Requirements Discussed:
- Migrate from monolithic architecture to microservices
- Complete redesign of customer-facing UI
- Implement new payment processing integration
- Add real-time inventory management
- Build admin dashboard for store management
- Implement recommendation engine
- Improve checkout experience

Technical Stack:
- Backend: Node.js/Express → FastAPI/Python
- Frontend: jQuery → React
- Database: PostgreSQL (migration needed)
        """,
        "tasks": [
            {
                "task_id": 1,
                "name": "Database Migration & Schema Design",
                "description": "Plan and execute migration from monolithic DB schema to microservices-compatible schema",
                "estimated_hours": 40,
                "estimated_cost_usd": 2000,
                "complexity": "High",
                "includes": [
                    "Schema analysis and redesign",
                    "Data migration scripts",
                    "Backup & rollback procedures",
                    "Performance optimization",
                    "Documentation"
                ]
            },
            {
                "task_id": 2,
                "name": "Product Microservice Implementation",
                "description": "Build standalone product service with catalog management, search, and filtering",
                "estimated_hours": 48,
                "estimated_cost_usd": 2400,
                "complexity": "High",
                "includes": [
                    "API endpoints (CRUD)",
                    "Search functionality",
                    "Filtering and sorting",
                    "Caching layer",
                    "Service documentation"
                ]
            },
            {
                "task_id": 3,
                "name": "Order Processing Microservice",
                "description": "Develop order management service with workflow processing and status tracking",
                "estimated_hours": 44,
                "estimated_cost_usd": 2200,
                "complexity": "High",
                "includes": [
                    "Order creation workflow",
                    "Status transitions",
                    "Order history tracking",
                    "Invoice generation",
                    "Order notifications"
                ]
            },
            {
                "task_id": 4,
                "name": "Payment Gateway Integration",
                "description": "Integrate modern payment processors (Stripe, PayPal) with secure transaction handling",
                "estimated_hours": 32,
                "estimated_cost_usd": 1600,
                "complexity": "High",
                "includes": [
                    "Stripe integration",
                    "PayPal integration",
                    "PCI compliance measures",
                    "Webhook handling",
                    "Error recovery"
                ]
            },
            {
                "task_id": 5,
                "name": "React Frontend Redesign",
                "description": "Build modern, responsive React application with improved UX and design",
                "estimated_hours": 56,
                "estimated_cost_usd": 2800,
                "complexity": "High",
                "includes": [
                    "Component architecture",
                    "State management (Redux)",
                    "Routing implementation",
                    "Form handling",
                    "Design system implementation"
                ]
            },
            {
                "task_id": 6,
                "name": "Inventory Management System",
                "description": "Real-time inventory tracking with stock alerts and automated reordering",
                "estimated_hours": 36,
                "estimated_cost_usd": 1800,
                "complexity": "High",
                "includes": [
                    "Inventory service",
                    "Stock tracking",
                    "Low-stock alerts",
                    "Reorder automation",
                    "Warehouse integration"
                ]
            },
            {
                "task_id": 7,
                "name": "Admin Dashboard Development",
                "description": "Build comprehensive admin interface for order and product management",
                "estimated_hours": 40,
                "estimated_cost_usd": 2000,
                "complexity": "Medium",
                "includes": [
                    "Dashboard layout",
                    "Data tables",
                    "Analytics widgets",
                    "Reporting features",
                    "User role management"
                ]
            },
            {
                "task_id": 8,
                "name": "Recommendation Engine",
                "description": "Implement ML-based product recommendations based on user behavior",
                "estimated_hours": 36,
                "estimated_cost_usd": 1800,
                "complexity": "High",
                "includes": [
                    "Algorithm development",
                    "User behavior analysis",
                    "Recommendation API",
                    "A/B testing framework",
                    "Performance tuning"
                ]
            },
            {
                "task_id": 9,
                "name": "Testing & QA",
                "description": "Comprehensive testing across all services and UI components",
                "estimated_hours": 48,
                "estimated_cost_usd": 2400,
                "complexity": "Medium",
                "includes": [
                    "Unit tests",
                    "Integration tests",
                    "E2E tests",
                    "Load testing",
                    "Security testing"
                ]
            },
            {
                "task_id": 10,
                "name": "Deployment & DevOps Setup",
                "description": "Setup CI/CD pipelines, containerization, and production deployment infrastructure",
                "estimated_hours": 32,
                "estimated_cost_usd": 1600,
                "complexity": "High",
                "includes": [
                    "Docker containerization",
                    "Kubernetes setup",
                    "CI/CD pipeline (GitHub Actions)",
                    "Monitoring & logging",
                    "Auto-scaling configuration"
                ]
            }
        ],
        "summary": {
            "total_hours": 412,
            "total_cost_usd": 20600,
            "team_size": "5 developers + 1 DevOps engineer",
            "estimated_duration_weeks": 12,
            "hourly_rate": 50
        }
    }
]

# JSON format example of a completed Hyrox workout for reference
WORKOUT_DATA_EXAMPLE = {
    "workout_id": "hyrox_20250914_001",
    "user_id": "user_12345",
    "workout_date": "2025-09-14",
    "workout_type": "Hyrox Competition",
    "total_duration_minutes": 78,
    "total_distance_km": 8.0,
    "average_heart_rate_bpm": 165,
    "max_heart_rate_bpm": 185,
    "min_heart_rate_bpm": 120,
    "total_calories_burned": 845,
    "average_calories_per_minute": 10.8,
    "heart_rate_zones": {
        "zone_1_recovery": {
            "name": "Recovery (50-60% max HR)",
            "min_bpm": 95,
            "max_bpm": 114,
            "time_minutes": 5,
            "percentage_of_workout": 6.4
        },
        "zone_2_aerobic": {
            "name": "Aerobic (60-70% max HR)",
            "min_bpm": 114,
            "max_bpm": 133,
            "time_minutes": 12,
            "percentage_of_workout": 15.4
        },
        "zone_3_threshold": {
            "name": "Threshold (70-80% max HR)",
            "min_bpm": 133,
            "max_bpm": 152,
            "time_minutes": 28,
            "percentage_of_workout": 35.9
        },
        "zone_4_vo2_max": {
            "name": "VO2 Max (80-90% max HR)",
            "min_bpm": 152,
            "max_bpm": 171,
            "time_minutes": 22,
            "percentage_of_workout": 28.2
        },
        "zone_5_anaerobic": {
            "name": "Anaerobic (90-100% max HR)",
            "min_bpm": 171,
            "max_bpm": 190,
            "time_minutes": 11,
            "percentage_of_workout": 14.1
        }
    },
    "stations": [
        {
            "station_number": 1,
            "station_name": "SkiErg",
            "time_seconds": 180,
            "completed": True
        },
        {
            "station_number": 2,
            "station_name": "50m Run",
            "time_seconds": 45,
            "completed": True
        },
        {
            "station_number": 3,
            "station_name": "WallBalls",
            "time_seconds": 150,
            "completed": True
        },
        {
            "station_number": 4,
            "station_name": "50m Run",
            "time_seconds": 48,
            "completed": True
        },
        {
            "station_number": 5,
            "station_name": "Rowing",
            "time_seconds": 165,
            "completed": True
        },
        {
            "station_number": 6,
            "station_name": "50m Run",
            "time_seconds": 46,
            "completed": True
        },
        {
            "station_number": 7,
            "station_name": "Sled Push",
            "time_seconds": 140,
            "completed": True
        },
        {
            "station_number": 8,
            "station_name": "50m Run",
            "time_seconds": 44,
            "completed": True
        },
        {
            "station_number": 9,
            "station_name": "Burpee Broad Jumps",
            "time_seconds": 120,
            "completed": True
        },
        {
            "station_number": 10,
            "station_name": "50m Run",
            "time_seconds": 42,
            "completed": True
        }
    ],
    "personal_metrics": {
        "age": 32,
        "weight_kg": 78,
        "height_cm": 182,
        "gender": "male",
        "max_heart_rate_measured": 190
    },
    "notes": "Excellent performance today. Strong finish on final run.",
    "created_at": "2025-09-14T19:45:00Z",
    "updated_at": "2025-09-14T21:30:00Z"
}

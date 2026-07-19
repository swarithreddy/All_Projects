# Professional IEEE Student Chapter Website

A responsive single-page website for the IEEE VJIT Student Branch. The project showcases the branch’s mission, chapter activities, events, team members, gallery highlights, and contact information in a polished, modern web experience.

## Features

- Responsive navigation bar with a mobile menu
- Hero section highlighting the IEEE VJIT Student Branch
- About section with the branch mission, vision, and objectives
- IEEE SSIT and IEEE CS chapter information with tabbed navigation
- Announcements cards with expandable content
- Upcoming and past events sections with filtering
- Team profiles for faculty leadership and chapter members
- Gallery section with category filtering and lightbox viewing
- Contact form with client-side submission feedback
- Smooth scrolling and scroll-to-top interaction

## Tech Stack

- Programming language: HTML, CSS, JavaScript
- Styling: Custom CSS with responsive layout rules
- Front-end interactivity: Vanilla JavaScript
- Typography: Google Fonts (Montserrat and Open Sans)
- Media: External images from Unsplash
- Build tools: Not found in the repository
- Package manager: Not found in the repository
- Backend / database: Not found in the repository

## Project Structure

```text
.
├── index.html          # Main landing page for the website
├── index1.html         # Additional HTML file present in the repository
├── styles.css          # Main stylesheet for the website layout and visuals
└── script.js           # Client-side behavior for navigation, tabs, filters, and forms
```

### File purposes

- index.html: Main homepage content for the IEEE VJIT Student Branch website
- index1.html: Another HTML file present in the repository; no additional documentation or references were found for its purpose
- styles.css: Contains the visual design system, layout rules, responsive styles, and section-specific styling
- script.js: Adds interactivity such as mobile menu toggling, tab switching, event filtering, gallery lightbox behavior, and contact form feedback

## Prerequisites

- A modern web browser such as Chrome, Edge, Firefox, or Safari
- No additional runtime or package installation is required to view the site

## Installation

1. Clone the repository to your local machine.
2. Open the project folder in your preferred editor or file explorer.
3. Open index.html in a web browser to view the website.
4. If you prefer to serve it over HTTP, use any simple static file server of your choice.

No dependency installation or build step was found in the repository.

## Configuration

No configuration files were found in the repository.

### Environment variables

No .env.example file or environment variable references were found in the repository.

## Running the Project

### Run locally

Open index.html directly in a browser.

### Development mode

The project is a static site, so there is no separate development server or dev build process configured in the repository.

### Production mode

No production build or deployment configuration was found in the repository.

### Build the project

No build step is defined for this project.

### Run with Docker

No Dockerfile or Docker Compose configuration was found in the repository.

## Usage

Once the site is opened in a browser, users can:

- Navigate between Home, About, Chapters, Events, Team, Gallery, and Contact sections
- Explore IEEE SSIT and IEEE CS chapter details
- Review upcoming and past events
- Browse the gallery and open images in the lightbox
- Use the contact form to send a message (the current implementation shows a browser alert rather than submitting to a backend)

## API Documentation

Not found in the repository.

This project does not expose backend endpoints or a REST API.

## Architecture

The website follows a simple static front-end architecture:

- HTML provides the structure and content for each section of the page
- CSS handles styling, layout, spacing, colors, and responsiveness
- JavaScript adds interactive behavior to support navigation, filtering, modal-style gallery viewing, and the contact form experience

The design is fully client-side and does not depend on a server-side framework or database.

## Important Modules

- Navigation and mobile menu: Handles section navigation and responsive menu behavior
- Chapter tabs: Allows users to switch between IEEE SSIT and IEEE CS chapter content
- Announcements: Expands and collapses announcement details
- Events filters: Toggles between all, SSIT, and CS event categories
- Gallery lightbox: Opens selected images in an overlay view
- Contact form: Collects user input and shows a client-side confirmation message

## Database

Not found in the repository.

No database configuration, migrations, or schema files are present.

## Scripts

No package.json, Makefile, or other script runner files were found in the repository.

## Testing

No automated tests or testing framework were found in the repository.

## Deployment

No deployment configuration, CI/CD workflow, or hosting configuration was found in the repository.

This project appears to be a static website intended for direct browser access or simple static hosting.

## Troubleshooting

- If the page does not render correctly, make sure all project files are present in the same directory.
- If styles appear missing, confirm that styles.css is available alongside index.html.
- If the JavaScript interactions do not work, check that script.js is present and that the browser allows local script execution.
- If images do not appear, verify that the browser has internet access for the external image URLs used in the page.

## Contributing

Contributions are welcome. A typical workflow is:

1. Fork the repository
2. Create a new branch for your changes
3. Make your updates and verify them in a browser
4. Submit a pull request describing the changes you made

Because this repository does not include automated tests or build scripts, visual verification in the browser is the main validation path.

## License

No license file was found in the repository.

## Acknowledgements

- IEEE VJIT Student Branch
- IEEE SSIT and IEEE CS chapters
- Google Fonts
- Unsplash for image assets

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add active class to navigation links based on scroll position
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-links a');
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (pageYOffset >= sectionTop - 60) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').slice(1) === current) {
            link.classList.add('active');
        }
    });
});

// Add animation on scroll
const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
};

const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe all sections
document.querySelectorAll('section').forEach(section => {
    observer.observe(section);
});

// Tag-based filtering: show only posts matching selected tag in category
const tagElements = document.querySelectorAll('.tag');
const postElements = document.querySelectorAll('.post');
let activeTag = null;

tagElements.forEach(tagEl => {
  tagEl.addEventListener('click', function(e) {
    e.preventDefault();
    const tagName = this.textContent.trim();

    // Handle toggling
    if (activeTag === tagName) {
      // Clear filter
      activeTag = null;
      tagElements.forEach(t => t.classList.remove('selected'));
      postElements.forEach(p => p.style.display = '');
    } else {
      // Apply new filter
      activeTag = tagName;
      tagElements.forEach(t => t.classList.toggle('selected', t.textContent.trim() === tagName));
      postElements.forEach(p => {
        const catEl = p.querySelector('.post-meta .category');
        const postCat = catEl ? catEl.textContent.trim() : '';
        p.style.display = (postCat === tagName) ? '' : 'none';
      });
    }
  });
}); 